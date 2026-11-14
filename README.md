# nemar-process

NEMAR's own processing repository: the launcher that runs as a Tapis ZIP-runtime application, runtime images, eegprep recipes with signalJourney provenance, and the Tapis definitions and queue observer.
It is Python only; there is no MATLAB anywhere in it.

- Operator and developer documentation is in [`docs/`](docs/): start with [`docs/launcher.md`](docs/launcher.md); the runtime image, its digests and how to verify it are in [`docs/images.md`](docs/images.md); the recipes and the `nemar-recipe` command are in [`docs/recipes.md`](docs/recipes.md).
- Development rules for people and coding agents are in [`AGENTS.md`](AGENTS.md).
- Released under the MIT license (see [`LICENSE`](LICENSE)).
