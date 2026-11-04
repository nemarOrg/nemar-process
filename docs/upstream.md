# Upstream projects

This repository depends on two projects it does not own: eegprep (`sccn/eegprep`) and signalJourney (`signalJourney/signalJourney`).
The rule for both is the same: a gap is fixed upstream, never patched here, and anything this repository copies is copied byte for byte with its source recorded.

Abbreviations used here: JavaScript Object Notation (JSON), Python Package Index (PyPI).

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

This phase files no eegprep issue.
The recipes phases pin each eegprep function they use and file an upstream issue for every gap they find; their links belong here.

One thing seen while building the smoke fixture, not filed: `clean_flatlines` of eegprep 0.5.0 raises `AttributeError: 'numpy.ndarray' object has no attribute 'get'` as soon as it removes a channel from a set that `pop_loadset` returned with an empty `etc`, because `update_clean_channel_mask` calls `.get` on `EEG["etc"]`.
`pop_saveset` writes an empty `etc` as an empty matrix and `pop_loadset` reads it back as an array, so the failure follows a plain save and load; setting `EEG["etc"] = {}` after loading avoids it.
The fixture generator therefore gives `etc` a key, and a recipe that loads sets it did not write should expect the same.
