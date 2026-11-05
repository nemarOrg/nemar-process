# Upstream projects

This repository depends on two projects it does not own: eegprep (`sccn/eegprep`) and signalJourney (`signalJourney/signalJourney`).
The rule for both is the same: a gap is fixed upstream, never patched here, and anything this repository copies is copied byte for byte with its source recorded.

Abbreviations used here: JavaScript Object Notation (JSON), Python Package Index (PyPI), Secure Hash Algorithm 256 (SHA-256), random sample consensus (RANSAC), electroencephalography (EEG).

## signalJourney

**The packaging issue:** [signalJourney/signalJourney#6](https://github.com/signalJourney/signalJourney/issues/6), "Publish the validator to PyPI with the schema packaged" (filed 2026-11-04).

The validator is not on PyPI, and it looks for its schema relative to its own source tree, so an installed copy cannot find one.
Until that is fixed upstream, this repository does two things:

1. installs the validator from a pinned commit, `signaljourney-validator @ git+https://github.com/signalJourney/signalJourney@<commit>`, declared once in the dependency group `signaljourney` of `pyproject.toml` and shared by the `dev` and `runtime` groups;
2. vendors that commit's `schema/` directory into `recipes/nemar_process/_vendor/signalJourney/schema/`, with the upstream `LICENSE` (BSD 3-Clause) and a `SOURCE` file naming the repository, the full commit and a digest of the copied tree, and points the validator at it: `nemar_process.sj.validator()` returns `Validator(schema_dir=<the vendored directory>)`.

`validator()` refuses to start when the vendored directory holds no schema version, because the validator's own fallback for an empty directory is the directory beside its source, which would validate against a schema this repository never reviewed.
It also captures the validator's progress lines instead of printing them, and fails if the validator reports an unresolvable `$ref`.

Two tests keep the copy honest.
`tests/images/test_sj.py` recomputes the digest of the vendored tree and compares it with `SOURCE`, so no file can change without the record changing.
In CI a second test compares every vendored file with a fresh checkout of signalJourney at the recorded commit (`NEMAR_SIGNALJOURNEY_DIR`, required there by `NEMAR_REQUIRE_SIGNALJOURNEY=1`); locally it runs when a checkout sits beside this one.

### What the `nemar-floor` recipe needed from signalJourney

The recipe's records validate against the vendored schema, and three gaps in it are filed (phase 3, 2026-11-04).
Each is worked around inside the recipe, and the workaround is named beside the code.

| Issue | Gap | The workaround here |
|---|---|---|
| [signalJourney/signalJourney#7](https://github.com/signalJourney/signalJourney/issues/7) | no checksum (or size) on output and input targets that reference files | `extensions.nemar.outputs` and `inputs` carry the size and SHA-256 of every file |
| [signalJourney/signalJourney#8](https://github.com/signalJourney/signalJourney/issues/8) | the `nemar` extension namespace is an empty open object | `extensions.nemar` has two profiles, `recording` and `dataset` (see [recipes.md](recipes.md)) |
| [signalJourney/signalJourney#9](https://github.com/signalJourney/signalJourney/issues/9) | the validator accepts malformed timestamps, non-finite numbers and dangling references | `records.check` tests them (timestamps, references) and the writer refuses NaN and infinity |

### Moving the commit

```bash
SJ=/path/to/signalJourney               # a checkout that contains the new commit
uv run python scripts/vendor_signaljourney.py --source "$SJ" --commit <new commit>
# change the commit in the `signaljourney` group of pyproject.toml, then:
uv lock
uv run python scripts/vendor_signaljourney.py --source "$SJ" --check
uv run pytest tests/images
```

The two commits (the group's and `SOURCE`'s) are compared by `tests/images/test_dependencies.py`.
Re-copy `tests/fixtures/sj/` from the new `schema/examples/` if the examples changed.
When upstream publishes a release with the schema packaged, drop the vendored copy, depend on the release and delete `scripts/vendor_signaljourney.py`.

## eegprep

The recipes phases pin each eegprep function they use and file an upstream issue for every gap they find; their links belong here.
Where a recipe works around a gap, the code names the issue beside the workaround, and a fix upstream makes the workaround unnecessary.

**[sccn/eegprep#533](https://github.com/sccn/eegprep/issues/533)**, "clean_flatlines fails with AttributeError on a dataset loaded with an empty etc" (filed 2026-11-04, found while building the smoke fixture).
`clean_flatlines` of eegprep 0.5.0 raises `AttributeError: 'numpy.ndarray' object has no attribute 'get'` as soon as it removes a channel from a set that `pop_loadset` returned with an empty `etc`, because `update_clean_channel_mask` calls `.get` on `EEG["etc"]`.
`pop_saveset` writes an empty `etc` as an empty matrix and `pop_loadset` reads it back as an array, so the failure follows a plain save and load; setting `EEG["etc"] = {}` after loading avoids it.
The workaround stays until that is fixed upstream: `scripts/make_smoke_fixture.py` gives the fixture's `etc` a key, with a comment pointing here, and a recipe that loads sets it did not write should expect the same.

### Gaps found by the `nemar-floor` recipe (phase 3, 2026-11-04)

Every reproduction was run against eegprep 0.5.0; the first and the eighth were also read on the `develop` branch of the same day.
Public wording only: "a processing pipeline wants X".

| Issue | Gap | What the recipe does meanwhile |
|---|---|---|
| [sccn/eegprep#533](https://github.com/sccn/eegprep/issues/533) (a comment adds the other failures) | an `etc` that is empty after a save and load is an array, and `clean_drifts`, `clean_windows`, `clean_flatlines`, `clean_channels`, `clean_channels_nolocs` and `pop_load_frombids` of a set all fail on it | the loader replaces a non-dict `etc` with `{}` and drops stale cleaning keys (`common/io.py`) |
| [sccn/eegprep#534](https://github.com/sccn/eegprep/issues/534) | no adaptive line-noise removal (no ZapLine, no CleanLine) | a fixed band-stop through `pop_eegfiltnew`; adopting ZapLine-plus later is a recipe version bump |
| [sccn/eegprep#535](https://github.com/sccn/eegprep/issues/535) (a comment adds `clean_flatlines`) | `clean_channels`, `clean_channels_nolocs` and `clean_windows` return the result and not what they measured, and the two channel functions return different shapes; `clean_flatlines` removes nothing when every channel has a flat portion and says so only in a log, with no mask | masks only; the table has no per-channel score, no per-window z-score and no per-criterion reason, and a recording with every channel flat is reported as fully good (docs/recipes.md, known limits) |
| [sccn/eegprep#536](https://github.com/sccn/eegprep/issues/536) | no public spectrum or line-noise helper and no robust-amplitude helper | `common/qc.py` calls `compute_spectra` and does NumPy arithmetic on its output |
| [sccn/eegprep#537](https://github.com/sccn/eegprep/issues/537) | `pop_eegfiltnew` does not expose the filter design it applied | `common/plan.py` reads the order, the transition and the cutoffs from the private design function and a test pins its answer |
| [sccn/eegprep#538](https://github.com/sccn/eegprep/issues/538) | `pop_load_frombids` leaves `None` in channel fields that `pop_saveset` cannot write | none needed (a recipe never saves a set); test code avoids saving what the loader returns |
| [sccn/eegprep#539](https://github.com/sccn/eegprep/issues/539) | `clean_channels` accepts degenerate channel locations and silently removes a large part of the channels | a rank test of the position vectors in `common/qc.py` before the call |
| [sccn/eegprep#540](https://github.com/sccn/eegprep/issues/540) | the RANSAC seed (5489) is hardcoded in `clean_channels` | recorded as an engine constant, and a test compares it with eegprep's source |
| [sccn/eegprep#541](https://github.com/sccn/eegprep/issues/541) | the cleaning steps differ in whether they mutate their input or return a copy | every step rebinds its working copy and never relies on the answer |

Three imports of the recipes reach into eegprep modules that are not documented API (`eegprep.functions.sigprocfunc.spectopo.compute_spectra`, `eegprep.plugins.firfilt._filtering.design_eegfiltnew` and `eegprep.plugins.EEG_BIDS.coords.chanloc_has_coords`).
A new eegprep can move them.
They are in two files (`common/qc.py` and `common/plan.py`), the lock pins the version, and one test per import asserts the call and its answer, so a move fails by name.
