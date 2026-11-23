# Standalone mode: `nemar-launch-local`

`nemar-launch-local` runs one job on a person's own Linux machine, from a local launch specification file and local directories, with no NEMAR service involved (ADR 0134 decision 8).
It is the launcher's second entry, shipped in the same zip as `tapisjob_app.sh` ([launcher.md](launcher.md)), and it runs the job's container exactly as NEMAR's service path does.
It never reads a NEMAR launch specification or a job token, calls no NEMAR route and uploads nothing.

Abbreviations used here: JavaScript Object Notation (JSON), American Standard Code for Information Interchange (ASCII), Secure Hash Algorithm 256 (SHA-256), Singularity Image Format (SIF), Open Container Initiative (OCI), Coordinated Universal Time (UTC).

## Running

```bash
unzip nemar-launch-<version>.zip -d nemar-launch
nemar-launch/nemar-launch-local \
  --spec job.json --out /path/to/out \
  --data ds003645=/path/to/ds003645 \
  --code code.tar.gz \
  --engine apptainer --sif-root /path/to/images
```

| Option | Meaning |
|---|---|
| `--spec FILE` | the local launch specification (below); required |
| `--out DIR` | the output directory; `work/` and `launcher/` are created in it and must not exist yet; required |
| `--data NAME=DIR` | a dataset the specification names, bound read-only, whole, at `/data/NAME`; repeat once per name |
| `--code FILE` | the code archive (`.tar.gz`) the specification's `bundle` names by SHA-256 and size |
| `--engine apptainer\|docker` | the container runtime; `apptainer` by default |
| `--sif-root DIR` | where `<64 hex>.sif` is (Apptainer; required with it) |
| `--cpus N`, `--memory-mb N` | the container's limits; both required with Docker, optional with Apptainer |
| `--scratch DIR` | where the code and the container's `/tmp` go (default: `TMPDIR`); removed at the end |
| `--time-limit-s N` | stop the job after N seconds (`time_limit`, exit 15) |
| `--allow-registry HOST` | a registry a Docker image may come from (default `ghcr.io`) |
| `--pull` | verify NEMAR's signed image with cosign and pull it if it is missing (below) |
| `--cosign PATH` | the cosign binary for `--pull` (default: `cosign` on `PATH`) |

`Ctrl-C` (SIGINT), a closed terminal (SIGHUP) and SIGTERM each stop the job, as SIGTERM stops a service job: the record says `terminated`, and the container's process group is killed if it has not stopped 10 seconds later.

The exit status is the job's: 0, or 10 to 16 for the failure code in `result.json` (the codes of [launcher.md](launcher.md)).
2 is a command line that cannot be used (a missing option, Docker without limits, a `--code` that is not a `.tar.gz`, a `--data` name given twice); then nothing is created.
An output directory that already holds `work/` or `launcher/` is `runtime_error`, `output_dir_not_empty` (exit 13), and nothing is written into it.

Linux with Apptainer, or with rootful Docker, is supported and tested.
Docker Desktop on macOS is not verified ([unverified.md](unverified.md)).

## The local launch specification

`nemar.local-launch.v1` is the launch specification without what only NEMAR's service gives: no `job_id`, `api_base`, `job_token` or `limits`.
`runtime`, `entry`, `command`, `args`, `bundle`, `image`, `env` and `resources` are the launch specification's own members and rules ([launcher.md](launcher.md), "The document"); it adds a required `format` and `datasets` by name only:

```json
{"format":"nemar.local-launch.v1","runtime":"python","entry":"run.py","args":[],"bundle":{"sha256":"<64 hex>","size":1234},"datasets":[{"name":"ds003645"}],"image":{"uri":"ghcr.io/nemarorg/nemar-python@sha256:<64 hex>","digest":"sha256:<64 hex>","sif":"<64 hex>.sif"},"env":{},"resources":{"gpus":0}}
```

The schema is the byte copy of nemar-cli's (`launcher/schema/compute-local-launch.v1.json`, [contracts.md](contracts.md)).
The file is read under the launch specification's encoding rules: printable ASCII only (no line break, not even a final one), at most 64 KiB, no duplicate key, no number with a fraction.
`jq -cj . job.json > spec.json` writes one so.
Its digest, recorded as `launch_spec_digest`, is the SHA-256 of the file's bytes.
A specification the launcher refuses is `runtime_error`, `invalid_spec` (exit 13), as on the service path.

The names on the command line are exactly the specification's (`invalid_spec` otherwise).
A dataset directory is resolved to its real path and bound whole: a directory that is missing is `dataset_fetch_failed`, `dataset_not_found` (exit 11); one that overlaps the output directory (inside it or holding it), or that is or holds the scratch directory, is `runtime_error`, `bind_source_unsafe`; a dataset inside the scratch directory is accepted.
Reading a dataset as a local Zarr store is a recipe follow-up, not part of this mode.

The code is the file given with `--code`: it is copied into the scratch while it is hashed, must hold exactly `bundle.size` bytes (`size_mismatch`) whose SHA-256 is `bundle.sha256` (`digest_mismatch`), and is then extracted with every rule of a fetched bundle; a file that cannot be read is `content_missing` (each `code_fetch_failed`, exit 10).

## What is the same as the service path, and what differs

The container phase is one piece of code for both entries (`__main__._do_work`): the runtime checks, the PID namespace probe, the bind table and the command line (`run.build_argv`, `runtime.build_argv`), the parent-death signal, the end time, the read-back of `work` (`run.finalize_outputs`), the output caps, and `result.json`, `provenance.json` and `manifest.json`.
`tests/launcher/test_local_argv.py` holds the two command lines equal for the same job.
What differs, and only this:

- Docker runs with `--pull=never` (the service path's is `--pull=missing`), and each run gets a random runner label (`nemar.runner=rnr_<24 hex>`); the job label names the placeholder job id `job_000000000000000000000000`.
- No device decision is made (as on Tapis): `resources.gpus` is what the specification says.
- The documents name the placeholder job id, `staging` is empty and `slurm_job_id` is null.
- The launcher's environment is built from an allowlist (`PATH`, `HOME`, `USER`, `LOGNAME`, `LANG`, `LC_ALL`, `LC_CTYPE`, `TMPDIR`, `XDG_RUNTIME_DIR`) and the `NEMAR_*` values derived from the command line, which pass through `settings.read_settings` as a runner's do.
  `NEMAR_LAUNCH_SPEC`, `NEMAR_LEASE_ID`, `_tapis*`, `SLURM_*`, `NEMAR_LOCAL_DATASETS_ROOT` and the device variables in the person's shell are never read.
- Nothing is uploaded, and live progress is never posted.

## The fence

The launcher has no reason to reach the network or start any program but the container runtime in this mode, and its fence (`nemar_launch/fence.py`) holds it to that inside its own process, with one audit hook (PEP 578).
For the whole run (the progress thread, the container phase, the read-back and the reports included) it refuses every socket, every name and reverse lookup, every `urllib` request, every program but `apptainer`, `singularity` or `docker` with the verbs the launcher uses (`--version`, `buildcfg`, `exec`, `info`, `image inspect`, `container inspect`, `rm`, and `run` with `--pull=never`), and `os.system`, `os.exec*`, `os.spawn*` and `os.posix_spawn`.
A refusal fails the call like any the system denies, and the log's last line counts the refusals (`fence: refusals none` in a normal run).

The fence does not reach into a child process: the container runtime's own network (an Apptainer image pull, the Docker daemon) is outside it.
Without `--pull` the launcher starts no pull at all.

## `--pull`

Without `--pull` nothing is pulled or verified: Apptainer needs `<sif-root>/<hex>.sif` (`image_missing` otherwise), and under Docker an image that is not present is `image_missing`.

With `--pull` the image must be NEMAR's own, `ghcr.io/nemarorg/nemar-python@sha256:<the specification's digest>` (a third-party image must already be present, and `--pull` refuses it before anything starts: `image_unverified`, reason `reference_refused`).
Then, once, after the runtime's checks and before the code, the PID namespace probe, the datasets and any user code, the fence opens one window around the verification and the pull and closes it however they end:

- Docker: `cosign verify` with nemar-process's image workflow identity (the runner's policy, [images.md](images.md)) must name the digest in every entry; the image is then pulled by that digest (`docker pull --quiet`) only if it is absent.
- Apptainer: cosign verifies the signed SIF artifact `ghcr.io/nemarorg/nemar-python-sif:sha256-<hex>`; the launcher reads that artifact's manifest from the registry by the digest cosign named, with an anonymous token (the only addresses the window lets this process request); the manifest must hash to that digest and hold one layer; a missing file is pulled by the manifest digest into a hidden part file, which must hash to the layer before it is renamed to `<hex>.sif`, and a file already there must hash to it too.

The window allows exactly the command lines computed when it opens (`cosign version --json`, `cosign verify ...`, the `docker pull`, and the `apptainer pull` once cosign has passed) and, under Apptainer, a lookup of the registry's host and connections to public addresses only after it.
cosign (3 or later) runs with `PATH` and a `HOME` inside the scratch, removed at the end, so the Sigstore trust root is fetched on every `--pull`; nothing is cached.
A failed verification is `runtime_error`, `image_unverified` (exit 13), with the reason in the log (`reference_refused`, `cosign_missing`, `cosign_failed`, `cosign_timeout`, `digest_mismatch`, `manifest_mismatch`, `layer_mismatch`); a pull that fails is `image_missing`.
Nothing runs after either.

## What is left in `<out>/launcher/`

| File | What |
|---|---|
| `result.json`, `provenance.json`, `manifest.json`, `launcher.log` | as on the service path ([launcher.md](launcher.md)) |
| `progress.json` | the live progress object (`nemar.progress.v1`), read from the recipe's progress file every 0.1 seconds until the first read, then every 10 seconds, and built once more after the container exited, with nothing running |
| `results-report.json` | the results report (`nemar.results-report.v1`), below |

The results report says what ran, on what and what came out: the run's random id and its UTC start and end, the launcher's version, the specification's SHA-256, the image with how it was verified (`null` without `--pull`), the code archive's SHA-256, each dataset's file count and bytes, a copy of `result.json`, the SHA-256 and size of each launcher file, the outputs' count, bytes and manifest digest, every recordings table of the manifest (read with the rules for a file the container wrote; its SHA-256 must equal the manifest's, or it is not read and says `digest_mismatch`), every signalJourney record (`dataset` beside a recordings table, `recording` otherwise), and, when the job wrote a recipe progress file, the checkpoint counts of its recordings by the backend's reconcile rules (`checkpoints`), else the reason there are none (`checkpoints_reason`).
Paths are relative to the job's outputs; no host path is in it.
It is checked against the copied schema (`launcher/schema/compute-results-report.v1.json`) and the rules a schema cannot state before it is written; a problem is a fixed word inside it and never changes the exit.
A run so short that the reporter never read the recipe's progress file before the recipe removed it has `checkpoints: null` with the reason `no_progress_file`.
A specification that could not be read leaves no report (there is no image to name), and the log says so.
