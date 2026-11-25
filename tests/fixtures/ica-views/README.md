# nemar-ica's views: a product with every view

`product/` is what `nemar-recipe run nemar-ica` writes for the repository's EEG fixture (`tests/fixtures/eeg/dataset`: one recording, 22 channels, 282 seconds at 200 Hz), five iterations of pAMICA on the CPU with one thread, given `index.json`, the serving index the data plane lists for that recording, built from the commit the run read (`--source-commit 5555...`: a run uses an index of its own commit only).
The converter keeps 200 Hz (its ceiling is 250 Hz), so the index lists one EEG group, `eeg_200hz`, of 56,400 samples, the grid the floor read.
It holds every view nemar-ica declares: the recipe's `icmaps`, `icactivity` and `midsegment` (nemar-cli ADR 0137), then nemar-quality's `icclasses` and nemar-spectrum's `spectrum` ([docs/plugins.md](../../../docs/plugins.md)), with each file they name, the recording's signalJourney record, the recordings table, the dataset record and the dataset `dataQuality` document.

It is the source of the fixtures of nemar-cli's backend (phases B1 and B2, storing a run's quality results and serving the files its views name) and of the website's quality panel.
Five iterations give a decomposition that has not converged (`did_converge` is false): the product shows the shape of the files, not a decomposition anyone should read.
The run's identity is fixed: the dataset `on007823` at `v1.0.0`, the source commit `5555...`, the image digest `sha256:eeee...` and the run id `ica-views-e2e`; the records also hold the clock and the nemar-process version they were made at.

What a backend reads of it:

- the recording's `dataQuality` entry, projected from its record as `record_entry` in [`tests/e2e/test_nemar_ica_views_e2e.py`](../../e2e/test_nemar_ica_views_e2e.py) shows: the recipe's metrics (`extensions.nemar.metrics`), each plugin's that ran (its step's `qualityMetrics`), the views and the views skipped, and the SHA-256 of the record's bytes;
- the views of the serving copy (`icactivity`, `midsegment`) name the store as a dataset directory's run does: the group the index lists, `content_sha256` null and the profile `archive`, matched by label;
- `extensions.nemar.ica.serving_index` names the index and the row it read (or why it gave none: `not_in_serving_index`, `serving_index_stale`), and `extensions.nemar.ica.positions` where the layout's positions came from;
- every file a view names is an output of a step of the record, with its size and SHA-256 in `extensions.nemar.outputs`.

## Making it again

```bash
NEMAR_ICA_VIEWS_PRODUCT=tests/fixtures/ica-views uv run --frozen --group runtime --extra cpu pytest tests/e2e/test_nemar_ica_views_e2e.py
```

The test runs the command into a temporary directory and copies the product into `product/` (and the index beside it, when the directory is another one).
The numbers depend on the machine's floating-point libraries, so a run elsewhere writes other digits: the copy was made on an arm64 macOS machine with the lock's CPU environment.
The test holds the copy to a fresh run on everything that does not depend on them: the skeleton of every JSON file (every float masked, and the clocks, file digests and sizes, the release's nemar-process version, the interpreter's patch release, the thread pools, torch's local version tag and the counts ASR, pAMICA and ICLabel derive from floating-point sums), every TSV header, the plan hash, the plugins and their source trees, each view's store inputs, parameters, caption and base, the metric names and units, and the shape of the dataQuality document.
It also runs a backend's checks on the copy (`records.problems_of` on both records, the files each view names, the recording's dataQuality entry against the bundle and the record checks), so a change of the recipe, of a plugin or of a view fails there until the copy is made again.
