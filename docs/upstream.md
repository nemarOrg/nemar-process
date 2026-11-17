# Upstream projects

This repository depends on two projects it does not own: eegprep (`sccn/eegprep`) and signalJourney (`signalJourney/signalJourney`).
The rule for both is the same: a gap is fixed upstream, never patched here, and anything this repository copies is copied byte for byte with its source recorded.

Abbreviations used here: JavaScript Object Notation (JSON), Python Package Index (PyPI), Secure Hash Algorithm 256 (SHA-256), random sample consensus (RANSAC), electroencephalography (EEG).

## signalJourney

**The packaging issue:** [signalJourney/signalJourney#6](https://github.com/signalJourney/signalJourney/issues/6), "Publish the validator to PyPI with the schema packaged" (filed 2026-11-04, released as `signalJourney` 0.2.0 on 2026-11-14).

The release is a wheel that packages its schema.
`Validator()` loads that schema through `importlib.resources`, raises `SchemaResolutionError` at construction for a `$ref` it cannot resolve, and logs on the logger `signaljourney_validator` instead of printing.
This repository therefore does two things:

1. installs the release, `signalJourney==0.2.0`, declared once in the dependency group `signaljourney` of `pyproject.toml` and shared by the `dev` and `runtime` groups (the distribution is `signalJourney`, the import package is `signaljourney_validator`, the command is `signaljourney-validate`);
2. keeps no copy of the schema and pins its tree digest instead: `recipes/nemar_process/sj.py` holds the constant `SIGNALJOURNEY_SCHEMA_TREE_SHA256`, the digest of the 37 files of `signaljourney_validator/schema/` in that release (`nemar_process.sj.tree_sha256`: the lines `<path> NUL <file sha256> LF`, in the order of the paths' components, hashed again).

`nemar_process.sj.validator()` computes the digest of the installed schema before it builds the validator, and raises `RuntimeError` when it differs from the constant or when the validator names no schema version, before any record is validated.
A record is therefore never validated against a schema this repository has not reviewed, whatever the lock or the environment holds.
It never prints: the validator's progress goes to its logger.

`tests/images/test_sj.py` pins all of it: the installed release is 0.2.0, the constant is the digest of the installed schema, a constant changed by one character refuses to start, and a changed, added, removed or renamed file is a different digest.
The same file compares the two example records of `tests/fixtures/sj/` with the release's own, byte for byte.
The constant was first recorded for the copy of commit 169c3261 that this release replaced, and the 37 files of the release are byte for byte the same as that copy, so no byte of the schema changed with the move.

### What the `nemar-floor` recipe needed from signalJourney

The recipe's records validate against the packaged schema, and three gaps in it are filed (phase 3, 2026-11-04).
Each is worked around inside the recipe, and the workaround is named beside the code.

| Issue | Gap | The workaround here |
|---|---|---|
| [signalJourney/signalJourney#7](https://github.com/signalJourney/signalJourney/issues/7) | no checksum (or size) on output and input targets that reference files | `extensions.nemar.outputs` and `inputs` carry the size and SHA-256 of every file |
| [signalJourney/signalJourney#8](https://github.com/signalJourney/signalJourney/issues/8) | the `nemar` extension namespace is an empty open object | `extensions.nemar` has two profiles, `recording` and `dataset` (see [recipes.md](recipes.md)) |
| [signalJourney/signalJourney#9](https://github.com/signalJourney/signalJourney/issues/9) | the validator accepts malformed timestamps, non-finite numbers and dangling references | `records.check` tests them (timestamps, references) and the writer refuses NaN and infinity |

### Moving the release

```bash
# 1. change the pin in the `signaljourney` group of pyproject.toml, then:
uv lock
uv sync --frozen
# 2. recompute the digest of the schema the new release packages:
PYTHONPATH=recipes uv run python -c "import importlib.resources as r; from nemar_process.sj import tree_sha256; print(tree_sha256(r.files('signaljourney_validator') / 'schema'))"
# 3. review what changed in the schema, in a checkout of signalJourney (tags v<old> and v<new>):
git diff v<old> v<new> -- src/signaljourney_validator/schema
# 4. change SIGNALJOURNEY_SCHEMA_TREE_SHA256 in recipes/nemar_process/sj.py to the digest of step 2, then:
uv run pytest tests/images
```

The pin, the lock, the reviewed schema difference and the constant move in the same pull request.
`uv lock` also rewrites the version line of `nemar-process` itself (`0.1.1-dev3` as `0.1.1.dev3`), which only the release bot changes (`docs/release.md`): put that one line back as it was.
A changed schema can change what a record needs, so look at `tests/recipes` (every record still validates) before the constant moves, and re-copy `tests/fixtures/sj/` from the new `schema/examples/` if the examples changed.

## eegprep

The recipes phases pin each eegprep function they use and file an upstream issue for every gap they find; their links belong here.
Where a recipe works around a gap, the code names the issue beside the workaround, and a fix upstream makes the workaround unnecessary.

**[sccn/eegprep#533](https://github.com/sccn/eegprep/issues/533)**, "clean_flatlines fails with AttributeError on a dataset loaded with an empty etc" (filed 2026-11-04, found while building the smoke fixture).
`clean_flatlines` of eegprep 0.5.0 raises `AttributeError: 'numpy.ndarray' object has no attribute 'get'` as soon as it removes a channel from a set that `pop_loadset` returned with an empty `etc`, because `update_clean_channel_mask` calls `.get` on `EEG["etc"]`.
`pop_saveset` writes an empty `etc` as an empty matrix and `pop_loadset` reads it back as an array, so the failure follows a plain save and load; setting `EEG["etc"] = {}` after loading avoids it.
The commit of `develop` the images pin (below) reads an empty `etc` back as a dictionary, so for a set the gap is closed there: the loader's repair (`common/io.py`) is a no-op for such a set and stays for 0.5.0 and for any other `etc` that is not a dictionary.
The smoke fixture is still written with a key in its `etc` (`scripts/make_smoke_fixture.py`, with a comment pointing here) because it must also load under 0.5.0, and a recipe that loads sets it did not write should still expect the failure on that release.

### The commit the images pin (`nemar-ica`, 2026-11-15)

No eegprep release holds pAMICA's `pop_runica(icatype='pamica')` (sccn/eegprep#617), the ASR state (#621) or eegprep-lean (#616, and #626 and #627 for its reader of a store's electrode positions): PyPI's newest is 0.5.0 (2026-11-04), and `develop` still says `0.5.0`.
The images therefore take eegprep from the merge commit of #617 on `develop`, `edf694c8344b9c4c68672eac3acf9a819f1462a8`, and every record, the engine stamp and `nemar-recipe version` name it (`0.5.0+edf694c8344b9c4c68672eac3acf9a819f1462a8`).
A release that contains it moves the pin to that release, which drops the suffix and gives a new engine string.
Three behaviors of 0.5.0 that this repository's tests pinned have changed at that commit, and the tests were changed to say so:

- `pop_loadbv` and `pop_loadset` refuse a data or marker file named outside the file's directory (an absolute path or a `..`), with a `ValueError` that names the reference; 0.5.0 followed it. A symbolic link that leaves the dataset is still followed, so the recipes' own check stays.
- `pop_loadset` returns an empty `etc` as a dictionary (sccn/eegprep#533 is fixed for that case); the loader's repair of a non-dict `etc` stays, and its load step now records `etcNormalized` false for such a set where it recorded true.
- A set with no sampling rate raises a `ValueError` at load; it used to load and fail the first check after it with a `TypeError`.

One gap found by `nemar-ica`, filed upstream as sccn/eegprep issue 625 and never worked around: the Zarr reader in eegprep-lean (`to_eeg`) built label-only channel locations, and did not read the electrode positions, system and units that a store carries in its root attributes, so ICLabel could not classify a recording read from a store. Pull request 626 fixed it, pull request 627 then fixed three edge cases of that reading, and the images take eegprep-lean from its `develop` merge commit (`2736f99265f278ea9b9184a8b639a2e384f7859b`, which holds #626 and #616), so ICLabel classifies a recording read from a store that declares positions. That `pop_iclabel` raises a `ValueError` from `topoplot` ("no channel has a finite theta and radius") on channels with no locations is correct behavior, and a store that declares none, or whose unit or coordinate system the reader refuses, still ends at `iclabel` with the detail `iclabel_no_locations`.

### Gaps found by the `nemar-floor` recipe (phase 3, 2026-11-04 and 2026-11-05)

Every reproduction was run against eegprep 0.5.0; the first and the eighth were also read on the `develop` branch of the same day.
Public wording only: "a processing pipeline wants X".

| Issue | Gap | What the recipe does meanwhile |
|---|---|---|
| [sccn/eegprep#533](https://github.com/sccn/eegprep/issues/533) (a comment adds the other failures) | an `etc` that is empty after a save and load is an array, and `clean_drifts`, `clean_windows`, `clean_flatlines`, `clean_channels`, `clean_channels_nolocs` and `pop_load_frombids` of a set all fail on it | the loader replaces a non-dict `etc` with `{}` and drops stale cleaning keys (`common/io.py`) |
| [sccn/eegprep#534](https://github.com/sccn/eegprep/issues/534) | no adaptive line-noise removal (no ZapLine, no CleanLine) | a fixed band-stop through `pop_eegfiltnew`; adopting ZapLine-plus later is a recipe version bump |
| [sccn/eegprep#535](https://github.com/sccn/eegprep/issues/535) (a comment adds `clean_flatlines`) | `clean_channels`, `clean_channels_nolocs` and `clean_windows` return the result and not what they measured, and the two channel functions return different shapes; `clean_flatlines` removes nothing when every channel has a flat portion and says so only in a log, with no mask | masks only; the table has no per-channel score, no per-window z-score and no per-criterion reason; the floor listens to the `clean_flatlines` log warning and fails a recording with every channel flat (`all_channels_flat`) |
| [sccn/eegprep#536](https://github.com/sccn/eegprep/issues/536) | no public spectrum or line-noise helper and no robust-amplitude helper | `common/qc.py` calls `compute_spectra` and does NumPy arithmetic on its output |
| [sccn/eegprep#537](https://github.com/sccn/eegprep/issues/537) | `pop_eegfiltnew` does not expose the filter design it applied | `common/plan.py` reads the order, the transition and the cutoffs from the private design function and a test pins its answer |
| [sccn/eegprep#538](https://github.com/sccn/eegprep/issues/538) | `pop_load_frombids` leaves `None` in channel fields that `pop_saveset` cannot write | none needed (a recipe never saves a set); test code avoids saving what the loader returns |
| [sccn/eegprep#539](https://github.com/sccn/eegprep/issues/539) | `clean_channels` accepts degenerate channel locations and silently removes a large part of the channels | a rank test of the position vectors in `common/qc.py` before the call, with NumPy's default tolerance (see below) |
| [sccn/eegprep#540](https://github.com/sccn/eegprep/issues/540) | the RANSAC seed (5489) is hardcoded in `clean_channels` | recorded as an engine constant, and a test compares it with eegprep's source |
| [sccn/eegprep#541](https://github.com/sccn/eegprep/issues/541) | the cleaning steps differ in whether they mutate their input or return a copy | every step rebinds its working copy and never relies on the answer |
| [sccn/eegprep#561](https://github.com/sccn/eegprep/issues/561) | `pop_loadbv` and `pop_loadset` read the file a header names wherever it is: an absolute path, a `..` or a sub-directory for a BrainVision `DataFile` or `MarkerFile`, an absolute `data` or `datfile` for a set | `common/headers.py` reads the header first and refuses any recording that names a file other than its own companions, or whose files lead outside the dataset directory through a link (`load_failed`, detail `external_reference`); `tests/recipes/test_headers.py` pins what eegprep alone does, so a fix upstream shows |

### The Zarr reader (`nemar-floor --input zarr`, 2026-11-13)

The recipe reads the serving copy with eegprep-lean from the `develop` merge commit of sccn/eegprep#627, which holds #626 and #616, installed from a pinned commit (see [recipes.md](recipes.md)).
The first pinned commit refused the unit the live index uses for microvolts, the micro sign (`µV`), so every real store failed; the recipe renamed the label to `uV` until the commit 14ba46e8 of sccn/eegprep#616 accepted the micro sign and the Greek mu.
The pin is the merge commit of #627 (`2736f99265f278ea9b9184a8b639a2e384f7859b`; the merge commits of #626 and #616 were `e53d3aa748a8ebd9816110463485b1d8edf624e5` and `af9145121ed8a69eab86bc2f5c425a6bdcced5cc`), which contains that commit, the recipe passes the unit through unchanged, and a test reads one store per spelling and compares the values.

Patch note for the reader that sccn/eegprep#616 merged (still true at the merge commit of #627, the pin): `load_recording` checks the size the index declares against `max_bytes` before it reads, and the shape of what it read afterwards (a store shorter than its index is refused), but it should also check each array's `zarr.json` (shape, chunk and shard grid, data type) against the index's declared channels and samples before it decodes anything, so that a store whose metadata disagrees with its index is refused by the reader and not left to the caller's size bound.

The reader of a store's electrode positions that sccn/eegprep#626 merged had three edge cases, found by reading it against the platform's locations digest and fixed by sccn/eegprep#627, the pin; none was worked around here, and no recipe code depends on the old behavior.
What the pinned reader does now:

- An integer coordinate too large for a binary64 (any of at least 2^1024 minus 2^970, or an integer literal of more than 4300 digits, which made Python's own parse raise before any entry existed) is a channel without a position, as the definition says, because the store root's integers are parsed as binary64; the store is read in full and digested.
- Every zero coordinate is read as positive zero, whatever its sign or spelling (`-0`, `-0.0`, `0`), as the digest hashes it, so two stores that differ only in the sign of a zero hand eegprep the same values, the sign of `Y` and of `theta` included.
  The BIDS import still keeps a negative zero that a float column of an `electrodes.tsv` spells with a minus sign, so for such a file `Y` and `theta` can differ in sign between the two routes; the locations digest is not affected.
- A position map that is not an object is refused, as the definition refuses it, as `ElectrodePositionsNotAnObject` (a subclass of `IndexError_`), so a run records the reason by the class name, and an `attributes` that is empty or null is no attributes.

The rank test of the location guard (sccn/eegprep#539) uses NumPy's default tolerance, which is of the order of machine precision times the number of channels.
A montage that is flat to a picometer therefore has numerical rank 3 and passes, and `clean_channels` still returns a degraded answer for it: with the 19 positions of the reproduction in that issue flattened onto a plane (thickness 1e-9 mm to 0.1 mm), it removed 2 of 19 channels from data that lose none with the true positions.
Where the line between a montage and a nearly flat one lies is a question for the upstream check, which should define its tolerance, so it is the point to add to that discussion; the recipe does not invent a threshold of its own.

Three imports of the recipes reach into eegprep modules that are not documented API (`eegprep.functions.sigprocfunc.spectopo.compute_spectra`, `eegprep.plugins.firfilt._filtering.design_eegfiltnew` and `eegprep.plugins.EEG_BIDS.coords.chanloc_has_coords`).
A new eegprep can move them.
They are in two files (`common/qc.py` and `common/plan.py`), the lock pins the version, and one test per import asserts the call and its answer, so a move fails by name.
