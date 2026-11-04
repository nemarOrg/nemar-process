# The launcher: `nemar-launch`

`nemar-launch` is the program NEMAR runs on a compute node as a Tapis ZIP-runtime application.
It fetches a code bundle, stages datasets, runs the entry in a container isolated from the host, and leaves provenance, a result manifest and one fixed failure code in a directory the container never saw.
It then uploads the outputs and those files through NEMAR's outputs routes.
The node holds no storage credential.

Nothing the container writes is trusted: NEMAR accounts for a job from what the launcher records, outside the container.

## Contents of the zip

`scripts/build_launcher_zip.py` builds `dist/nemar-launch.zip`, byte for byte the same on every build.

```text
tapisjob_app.sh                  # the entry point Tapis runs
nemar_launch/*.py                # standard library only, Python 3.9 or newer
schema/launch-spec.v1.json       # the launch specification schema (see contracts.md)
```

```bash
uv run python scripts/build_launcher_zip.py            # write dist/nemar-launch.zip and print its sha256
uv run python scripts/build_launcher_zip.py --check    # verify determinism and layout, write nothing
```

## Running

Tapis unpacks the zip into the job's execution directory and runs `./tapisjob_app.sh` there.
The script:

1. finds the first of `python3.12`, `python3.11`, `python3.10`, `python3.9`, then `python3`, that is at least 3.9 (override with `NEMAR_HOST_PYTHON`, which must itself be 3.9 or newer);
2. runs the launcher in isolated mode (`python -I`), so the environment and the working directory cannot redirect it to other code;
3. writes the launcher's exit number to `${_tapisExecSystemOutputDir}/tapisjob.exitcode` and exits with it.

`python -I -m nemar_launch` cannot find a package that is only next to the script, because isolated mode ignores `PYTHONPATH` and the working directory.
The script therefore starts the same module with `runpy.run_module`, after putting the zip's own directory on the path.

A SIGTERM sent to the script (Slurm's, at the time limit or on cancel) is passed on to the launcher, which passes it to the container; the script still writes `tapisjob.exitcode` afterwards.
If no usable Python is found, or `_tapisExecSystemOutputDir` is unset, the script exits 13 with a message on stderr.

## The launch specification

The Worker hands the launcher one document, in the environment variable `NEMAR_LAUNCH_SPEC`: JSON, ASCII only (every character at or above U+007F is written as `\uXXXX`), at most 64 KiB.
`launcher/schema/launch-spec.v1.json` is its JSON Schema, shared byte for byte with nemar-cli (see [contracts.md](contracts.md)).
The launcher validates against it with a small validator of its own, because it may use only the standard library; a schema keyword the validator does not implement is an error, never ignored.

| Key | Meaning |
|---|---|
| `job_id` | the NEMAR job id, `job_` and 24 hex digits |
| `runtime` | `python` or `image`; anything else is `runtime_error` with detail `unsupported_runtime` |
| `entry` or `command` | exactly one: a path inside the bundle, or an argument vector for an image job |
| `args` | arguments, after the entry or the command |
| `bundle` | `{sha256, size}` of the code archive; required with `entry`, optional with `command`; no URL |
| `datasets` | `[{id, version, include?}]`, each id at most once; `include` is a list of globs |
| `api_base` | NEMAR's API; the launcher calls it with `job_token` |
| `image` | `{uri, digest, sif}`; `sif` is a relative path to the Singularity image file |
| `env` | the allowlisted environment for the container |
| `resources` | `{gpus}`; `--nv` only when above zero |
| `limits` | `{max_files, max_total_bytes}` for staged datasets |
| `job_token` | the per-job token, sent in the `X-Compute-Job-Token` header, never in a URL |

There are no unknown keys, at any level.
A specification that is not valid never reaches the network: the launcher writes its three files and exits 13 (`runtime_error`, detail `invalid_spec`).

The launcher hashes the raw bytes of the environment value before it parses them (`os.environb`), and records the SHA-256 as `launch_spec_digest` in `provenance.json`.
It is not the job row's `spec_digest`, which NEMAR computes over the job specification it stores.

`entry` runs as `python /code/<entry> <args>` for `runtime: python`, and as `/code/<entry> <args>` for `runtime: image`.
A `command` runs as given, followed by `args`.

## Layout

The output directory is `${_tapisExecSystemOutputDir}`, the directory Tapis archives.

```text
work/            the container's only writable mount, at /work
  outputs/       what the entry writes (NEMAR_OUT=/work/outputs)
  tmp/           mounted as /tmp
launcher/        written by the launcher, never mounted into the container
  result.json
  provenance.json
  manifest.json
  launcher.log
tapisjob.exitcode
```

Code and datasets stage in node-local scratch (`SLURM_TMPDIR`, else `TMPDIR`), never in the archived directory, and are removed when the run ends.
A scratch directory inside the output directory is refused.

## The container

`apptainer exec`, else `singularity exec`.
The arguments are built in one function (`run.build_argv`) and pinned by tests:

```text
exec --containall --cleanenv --no-home --pwd /work
     --bind <out>/work:/work  --bind <out>/work/tmp:/tmp
     [--bind <scratch>/code:/code:ro]  [--bind <dataset>:/data/<id>:ro ...]
     [--nv]  --env <name>=<value> ... --env NEMAR_OUT=/work/outputs
     <image> <entry or command> <args>
```

- The container's environment is the job's allowlisted variables and `NEMAR_OUT`, nothing from the host.
- The runtime process itself gets only `PATH`, `HOME`, `USER`, `LOGNAME`, `LANG`, `LC_ALL`, `LC_CTYPE`, `TMPDIR` and `XDG_RUNTIME_DIR`, because `APPTAINERENV_*` and `SINGULARITY_BIND` style variables in the launcher's environment would otherwise be honoured.
- `launcher/` is a sibling of `work/` and is not mounted, so the container can neither read nor forge the result, the provenance or the manifest.
- `/code` and `/data/<id>` are read-only.

The entry's standard output and error go to Tapis's `tapisjob.out`; the launcher's own log goes to `launcher/launcher.log` and to standard error.

## Bundle

`POST <api_base>/compute/jobs/<job_id>/bundle` with the token returns a ten-minute presigned address of `bundle.tar.gz`.
The launcher downloads it to scratch with the same retry, size and SHA-256 checks as a dataset file, then inspects every member of the archive before extracting any.
An archive is refused whole when a member is absolute, contains `..`, is a link, a device, a FIFO or another special file, appears twice, or has a file as a parent, or when it expands past 4 GiB or 100,000 members.
Extraction is hand-written, creates regular files and directories only, and removes the setuid, setgid, sticky and group and world write bits.
A refusal, a digest mismatch or a failed download is `code_fetch_failed`.

## Datasets

For each dataset the launcher reads `GET <api_base>/data/<id>/<version>/manifest.json` (entries `{path, size, checksum?, checksum_algorithm?, bytes_url, url}`, at most 38,000).
It selects the files the `include` globs name (`**` crosses directories, `*` and `?` stay within one, `[..]` is a class; no `include` selects all), and checks the selection against `limits` before fetching any byte.

- **bind**: when `NEMAR_LOCAL_DATASETS_ROOT` holds `<id>/<version>/` and every selected file there matches the manifest by size and checksum (`SHA256E` sha256, or the git blob SHA-1 for `git`), that directory is mounted read-only; no file is requested.
  A symlink, a missing file, a wrong size, a different byte or an entry without a checksum falls back to download.
- **download**: each file from its `bytes_url` into scratch, five attempts with backoff and jitter, `Range` resume after a truncated body, size and checksum checks, four files at a time.
  A 502 whose `error` is "Published file is missing from storage" is lost content: it is not retried and its path is logged.
  Any other 502 is a transport fault and is retried.

A manifest path that could leave the dataset directory is refused before anything is downloaded.
Every dataset failure is `dataset_fetch_failed`.

## After the entry exits

The launcher removes every symlink and special file under `work/` (and records each in the log), then hashes `work/outputs` into `manifest.json`: path, size and SHA-256, sorted by path.
A file whose name the documents cannot carry (a backslash, a control character, a name that is not UTF-8) is left out of the manifest and recorded in the log.

`provenance.json` holds the launch specification digest, the bundle digest, the image URI and digest, a free-text runtime version (the container runtime's own version line and the host Python), the staging records (`id`, `version`, `mode`, `files`, `bytes`), phase timings (`code_fetch`, `staging`, `run`, `finalize`), the Slurm job id and the entry's exit code.
`result.json` holds the failure code, a fixed-word detail, the exit number and the entry's exit code.
All three are validated against the schema's `$defs` in the tests.

## Upload

After the manifest is written the launcher uploads, with the token only on NEMAR's own routes:

1. `POST <api_base>/compute/jobs/<job_id>/outputs`: every output's path, size and SHA-256, and the sizes of the four launcher files; the answer is one presigned address per file.
2. A PUT of each file to its address: outputs first, four at a time; then `manifest.json`, `provenance.json`, `launcher.log` and, last, `result.json`.
   Three attempts each with backoff; one re-request of the addresses after a 403 for the whole run; no redirect is followed; the token is not sent.
3. `POST <api_base>/compute/jobs/<job_id>/outputs/complete`, once, after every PUT succeeded.
   `complete` is the commit: files without it are not a result.

`launcher.log` is uploaded as it was when its size was declared, although the launcher keeps logging.
A job that failed earlier still uploads what it made and the launcher files, and calls `complete`.

When the full upload fails (a refusal for a size cap or a quota, a PUT that fails past its retries, a refused `complete`), the launcher makes one more request that names no outputs and uploads only the four launcher files, so NEMAR can read why; `complete` is not called then.
The failure is `result_upload_failed`, with the refusal's own code word as the detail when there is one (`results_too_large`, `results_quota_exceeded`, `too_many_outputs`), and it never replaces an earlier failure code.

## Failure codes and exit numbers

Exactly these seven, in nemar-cli's order.
The first failure wins.

| Exit | Code | When |
|---|---|---|
| 0 | none | nothing failed |
| 10 | `code_fetch_failed` | the bundle route refused, the download failed, the digest or size differs, the archive is unsafe |
| 11 | `dataset_fetch_failed` | a manifest, file, checksum, size, path or cap failure while staging |
| 12 | `entry_not_found` | the entry is not a file in the bundle |
| 13 | `runtime_error` | the entry exited non-zero; an unsupported runtime or invalid specification; no container runtime, a missing image; a SIGTERM before the time limit (detail `terminated`) |
| 14 | `out_of_memory` | exit status 137 (SIGKILL) |
| 15 | `time_limit` | a SIGTERM at or after `SLURM_JOB_END_TIME` minus 120 seconds |
| 16 | `result_upload_failed` | the upload failed and nothing failed earlier |

The detail is a fixed word, never text from a response or the container.
Some details are `unsupported_runtime`, `invalid_spec`, `entry_failed`, `terminated`, `no_container_runtime`, `image_missing`, `scratch_in_output`, `scratch_unavailable`, `internal_error`, `digest_mismatch`, `size_mismatch`, `checksum_mismatch`, `content_missing`, `file_not_found`, `unsafe_archive`, `unsafe_path`, `too_many_files`, `too_many_bytes`, `bad_manifest`, `manifest_too_large`, `dataset_not_found` and `incomplete_plan`; an HTTP failure is `http_<status>`, `timeout` or `network_error`, or the response's own `code` word when it has one.

A signal death of the entry is reported as 128 plus the signal, as a shell does, in `entry_exit_code`.

An error the launcher did not expect (a bug) is also recorded, as `runtime_error` with detail `internal_error` (or `result_upload_failed` with the same detail during the upload), with only the exception's class name in the log, and the run still leaves its three files.

## Environment

| Variable | Who sets it | Meaning |
|---|---|---|
| `NEMAR_LAUNCH_SPEC` | the Worker, through Tapis | the launch specification; holds the job token |
| `_tapisExecSystemOutputDir` | Tapis | the output directory |
| `NEMAR_HOST_PYTHON` | the application definition, optional | the host Python to use |
| `NEMAR_LOCAL_DATASETS_ROOT` | the application definition, optional | root of NEMAR's own copy of datasets, `<root>/<id>/<version>/` |
| `NEMAR_SIF_ROOT` | the application definition | the directory `image.sif` is relative to; when unset, the working directory |
| `SLURM_TMPDIR`, `TMPDIR` | Slurm, the system | where scratch goes, in that order |
| `SLURM_JOB_END_TIME`, `SLURM_JOB_ID` | Slurm | the time limit rule, and the job id in provenance |

Tapis archives `tapisjob.env` by default and the specification carries the job token, so the launcher never prints the specification, the token or a presigned address.
Logs show a URL as host and path only, and never copy a response body.

## Developing

```bash
uv sync --frozen
uv run pre-commit install        # once per clone: ruff on the staged files, with --fix --unsafe-fixes
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest --cov=launcher --cov-fail-under=85
uv run --no-project --python 3.9 --with pytest --with jsonschema pytest tests/launcher
uv run python scripts/build_launcher_zip.py --check
```

The tests use real files, real processes and local HTTP stand-ins built on the standard library (`tests/standin/`): a data plane over `tests/fixtures/dataset/`, NEMAR's compute routes with an object-storage PUT receiver, and a stand-in for the container runtime binary for the tests that are about the launcher's flow.
No response in a stand-in was recorded from the real service; the shapes come from the contract files.

The container tests (marker `apptainer`) need a real Apptainer or Singularity binary.
They skip locally, with the reason stated, where there is none, and run in CI against a pinned Apptainer and a small public image pinned by digest.
In CI `NEMAR_REQUIRE_APPTAINER=1` turns a missing binary into a failure.
To run them on a Linux machine with Apptainer, `uv run pytest -m apptainer`; `NEMAR_TEST_SIF` names a prebuilt image to skip the pull.
