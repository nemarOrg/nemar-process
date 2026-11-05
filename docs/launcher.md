# The launcher: `nemar-launch`

`nemar-launch` is the program NEMAR runs on a compute node as a Tapis ZIP-runtime application.
It fetches a code bundle, stages datasets, runs the entry in a container isolated from the host, and leaves provenance, a result manifest and one fixed failure code in a directory the container never saw.
It then uploads the outputs and those files through NEMAR's outputs routes.
The node holds no storage credential.

Nothing the container writes is trusted: NEMAR accounts for a job from what the launcher records, outside the container.

Abbreviations used here: JavaScript Object Notation (JSON), American Standard Code for Information Interchange (ASCII), Secure Hash Algorithm 256 (SHA-256), Hypertext Transfer Protocol (HTTP) and its secure form (HTTPS), application programming interface (API), Uniform Resource Locator (URL), process identifier (PID), Singularity Image Format (SIF), graphics processing unit (GPU), comma-separated values (CSV), continuous integration (CI), out of memory (OOM), Unicode Transformation Format (UTF-8, UTF-16), Request for Comments (RFC), null character (NUL).
Cloudflare R2 object storage is called R2.

## Contents of the zip

`scripts/build_launcher_zip.py` builds `dist/nemar-launch.zip`, byte for byte the same on every build.

```text
tapisjob_app.sh                  # the entry point Tapis runs
lib/nemar_launch/*.py            # standard library only, Python 3.9 or newer
lib/schema/launch-spec.v1.json   # the launch specification schema (see contracts.md)
```

```bash
uv run python scripts/build_launcher_zip.py            # write dist/nemar-launch.zip and print its sha256
uv run python scripts/build_launcher_zip.py --check    # verify determinism and layout, write nothing
```

## Running

Tapis unpacks the zip into the job's execution directory and runs `./tapisjob_app.sh` there.
The script:

1. sets `umask 077`, so every file the launcher makes, and every file the container makes by default, is owner-only;
2. finds the first of `python3.12`, `python3.11`, `python3.10`, `python3.9`, then `python3`, that is at least 3.9, checking each with `python -I` (override with `NEMAR_HOST_PYTHON`, which must itself be 3.9 or newer);
3. runs the launcher with `python -I -S`: isolated mode (no `PYTHON*` variables, no working directory or script directory on the import path, no user site) and no `site` module (no `.pth` files, no `sitecustomize`), with ONLY `lib/` on the import path, so nothing Tapis or a job leaves beside the script can shadow a module;
4. writes the launcher's exit number to `${_tapisExecSystemOutputDir}/tapisjob.exitcode` and exits with it.

`python -I -m nemar_launch` cannot find a package that is only next to the script, because isolated mode ignores `PYTHONPATH` and the working directory.
The script therefore starts the same module with `runpy.run_module`, after putting `lib/` on the path.

A SIGTERM (the termination signal) sent to the script, which Slurm does at the time limit or on cancel, is passed on to the launcher, which passes it to the container's whole process group; the script still writes `tapisjob.exitcode` afterwards.
If no usable Python is found, or `_tapisExecSystemOutputDir` is unset, the script exits 13 with a message on stderr.

## The launch specification

The Worker hands the launcher one document, in the environment variable `NEMAR_LAUNCH_SPEC`.
The document is JSON text, printable ASCII only (every byte is 0x20 to 0x7E: a control character is a JSON escape, and every character at or above U+007F, DEL included, is a `\uXXXX` escape), at most 64 KiB.
`launcher/schema/launch-spec.v1.json` is its JSON Schema, shared byte for byte with nemar-cli (see [contracts.md](contracts.md)).
The launcher validates against it with a small validator of its own, because it may use only the standard library; a schema keyword the validator does not implement is an error, never ignored.

### The environment value: base64url

The variable does not hold the JSON text.
It holds that text encoded as base64url without padding: the 64 characters of section 5 of RFC 4648 (`A-Za-z0-9-_`), no `=`, no line break, so at most 87,382 characters (`ceil(65536 * 4 / 3)`) for a text of 64 KiB.
The text, its digest and its 64 KiB cap are unchanged in meaning; only the value that travels is encoded.

Why: the scheduler writes each environment value into a launch file that a shell sources on the node, and it quotes a value only when it holds a space, one of `&`, `>`, `<`, `|` or `;`, or a backtick, and then double-quotes it and escapes only the double quote (from the scheduler's source, to confirm on the tenant).
Compact JSON is not safe there.
A value with no space is left bare, so the shell removes its quotes.
A `$`, a backtick or a backslash in a user's string is expanded, runs a command or breaks the quoting.
The encoded value holds no character a shell treats as special, so it reaches the launcher unchanged whether or not the scheduler quotes it.
`tests/launcher/test_shell_quoting.py` implements that quoting rule, sources the launch file with each of `sh`, `dash` and `bash`, and shows both halves: the raw JSON form breaks (and runs a command a user put in a string), and the encoded form comes back exactly and runs nothing.

The decoder (`spec.decode_wire`) mirrors nemar-cli's `decodeLaunchSpecEnv` rule for rule.
It is strict and bounded, uses only the standard library, and checks the length before it reads a byte.
Every refusal is `runtime_error` with detail `invalid_spec`: there is no new failure code, and no older form is accepted, because nothing was ever registered with a tenant on one.
The log names the rule that fired, as one fixed word, and never any part of the value, which holds the job token once decoded.

The checks run in this order, and the first to fail names the word:

| # | The value is refused when | Word in the log |
|---|---|---|
| 1 | it is not bytes (an unset variable is `None`) | `not_base64url` |
| 2 | it is empty | `empty` |
| 3 | it is longer than 87,382 characters (judged by the length alone, before anything else is read) | `too_long` |
| 4 | it holds any character outside `A-Za-z0-9-_`: padding, `+` or `/` of the standard alphabet, whitespace, a line break, a quote, a raw JSON document | `not_base64url` |
| 5 | its length is 1 modulo 4, which no byte string encodes to | `bad_length` |
| 6 | once the `=` padding is restored by length (two signs for 2 modulo 4, one for 3, none for 0) and the value decoded, any byte is outside 0x20 to 0x7E: a control character, a tab or a newline, DEL, or anything at or above 0x80 | `not_printable` |
| 7 | it is not the canonical encoding of what it decodes to (encoding the decoded text again does not give the value back), which refuses non-zero trailing bits | `not_base64url` |

An accepted value decodes to printable ASCII, not empty, and at most 65,536 bytes.
`QQ` is accepted (it is `A`); `QR` decodes to the same byte and is refused, because it is not the value the encoder writes.
For a value with a character outside ASCII, `too_long` and `not_base64url` are both conforming answers, since languages count a length in different units (this launcher counts bytes); only an ASCII value has its word pinned.
A value that decodes to text which is not one JSON object, or which fails the schema, is refused by the rules below with the same detail and no word.

**Registration is still BLOCKED, and no longer by the encoding.**
Nothing is to be registered with any tenant until the owner items recorded on nemarOrg/nemar-process#9 are settled:
the release-asset procedure (immutable releases, tag rulesets and a reviewed release workflow),
whether the entry scripts delete the launch files after the job starts,
whether the launcher adds `--no-mount hostfs`,
and confirmation on the tenant of every Tapis behaviour marked "from source, to confirm on the tenant".
The encoding of the launch specification is no longer one of them:
the value of `NEMAR_LAUNCH_SPEC` is the base64url encoding (no padding) of the specification's JSON text,
so a shell that sources the launch file reads only letters, digits, `-` and `_`,
and the launcher decodes it strictly.
No job with a real token is to be submitted before then either ([unverified.md](unverified.md)).

### The document

| Key | Meaning |
|---|---|
| `job_id` | the NEMAR job id, `job_` and 24 hex digits |
| `runtime` | `python` or `image`; anything else is `runtime_error` with detail `unsupported_runtime` |
| `entry` or `command` | exactly one: a path inside the bundle, or an argument vector for an image job |
| `args` | arguments, after the entry or the command |
| `bundle` | `{sha256, size}` of the code archive; required with `entry`, optional with `command`; no URL |
| `datasets` | `[{id, version, include?}]`, each id at most once; `version` is the data-plane tag; `include` is a list of globs |
| `api_base` | NEMAR's API; the launcher calls it with `job_token`; HTTPS, or HTTP on loopback only |
| `image` | `{uri, digest, sif}`; `sif` is `<64 hex digits of the digest>.sif`, relative to `NEMAR_SIF_ROOT` |
| `env` | the allowlisted environment for the container |
| `resources` | `{gpus}`; `--nv` only when above zero |
| `limits` | `{max_files, max_total_bytes}` bound the dataset selection only (checked before any file is fetched); they do not bound the outputs |
| `job_token` | the per-job token, sent in the `X-Compute-Job-Token` header, never in a URL |

There are no unknown keys, at any level.
A specification that is not valid never reaches the network: the launcher writes its three files and exits 13 (`runtime_error`, detail `invalid_spec`).

**A dataset version is the data-plane tag, used verbatim.**
It is `latest` or `vN.N.N` (for example `v1.0.2`), the only two forms the data plane accepts; a bare `1.0.2` is a 404 there.
The Worker resolves whatever spelling the user typed into one of those two before it writes the specification, and the launcher refuses anything else as `invalid_spec`.
The local copy and the data plane's routes use the same string.

**The image file is named by the image digest.**
`image.sif` must equal the 64 hexadecimal digits of `image.digest`, then `.sif`, and `image.uri` must end with `image.digest`; any other value is `invalid_spec` (NEMAR's encoder never writes one).
The file is looked up under `NEMAR_SIF_ROOT` and nowhere else: an unset or empty `NEMAR_SIF_ROOT` is `runtime_error` with detail `image_missing` (there is no fallback to the working directory), and the resolved file must stay under the real root (no symlink out of it).
The launcher does not hash the image: only its name ties it to the digest.

The launcher reads the environment value as bytes (`os.environb`), decodes it, hashes the JSON text it decodes to before it parses that text, and records the SHA-256 as `launch_spec_digest` in `provenance.json`, even for a text it then refuses.
The digest is over the text and never over the encoded value, and it is not the job row's `spec_digest`, which NEMAR computes over the job specification it stores.
A value that does not decode to a text has no text to hash: its digest is that of no bytes (as when the variable is not set), and the job is unnamed.

`entry` runs as `python /code/<entry> <args>` for `runtime: python`, and as `/code/<entry> <args>` for `runtime: image`.
A `command` runs as given, followed by `args`.

## Layout

The output directory is `${_tapisExecSystemOutputDir}`, the directory Tapis archives.
The launcher creates `work`, `work/outputs` and `launcher` itself, with `os.mkdir`.
**If `work` or `launcher` already exists, whatever it is (a directory, a symlink), the launcher does not start**: it records `runtime_error` with detail `output_dir_not_empty`, writes nothing into the output directory, and uploads its files from memory only.
A planted `work/tmp -> ..` or `launcher -> /` could otherwise lead a write or a bind mount out of the archived directory.

```text
work/            the container's only archived mount, at /work
  outputs/       what the entry writes (NEMAR_OUT=/work/outputs)
launcher/        written by the launcher, never mounted into the container
  result.json
  provenance.json
  manifest.json
  launcher.log
tapisjob.exitcode
```

Code, datasets and the container's `/tmp` live in node-local scratch (`SLURM_TMPDIR`, else `TMPDIR`), never in the archived directory, and are removed when the run ends.
A scratch directory inside the output directory is refused.

**The container's `/tmp` is scratch, not `work/tmp`.**
The first design bound `work/tmp` at `/tmp`.
That puts every temporary file a job makes into the archived directory, where it counts against the shared account's quota, so `/tmp` is bound to a scratch directory instead.
This deviates from the bind table in the phase issue, on the advisor's recommendation; `work/` stays the only archived writable mount.

The launcher's own files are opened without following a link (`O_NOFOLLOW`), the log exclusively (`O_EXCL`) and each document through a temporary file created exclusively and renamed into place, so a link planted at a final name is replaced, never followed.

## The container

`apptainer exec`, else `singularity exec`.
The arguments are built in one function (`run.build_argv`) and pinned by tests:

```text
exec --containall --cleanenv --no-home --no-mount bind-paths --pwd /work
     --bind <out>/work:/work  --bind <scratch>/tmp:/tmp
     [--bind <scratch>/code:/code:ro]  [--bind <dataset>:/data/<id>:ro ...]
     [--nv]  --env <name>=<value> ... [--env CUDA_VISIBLE_DEVICES=<ids>]  --env NEMAR_OUT=/work/outputs
     <image> <entry or command> <args>
```

- **No site binds.** `--no-mount bind-paths` turns off the `bind path` lines of the runtime's own configuration, which would otherwise mount host directories into every container whatever the flags.
- **A PID namespace, proven or refused.** `--containall` gives the container a PID namespace of its own; a runtime whose configuration forbids it (`allow pid ns = no`) carries on without one, and user code could then read the launcher's environment, the job token, out of `/proc`. The launcher fails closed: before any user code runs it must verify the namespace, or it refuses with `runtime_error`, detail `pid_namespace_unavailable`. It reads the setting from the runtime's configuration. When `<runtime> buildcfg` names the runtime's own `SYSCONFDIR`, that directory is authoritative: only `<SYSCONFDIR>/<runtime>/<runtime>.conf` is read, and if it is missing the result is "cannot be verified" (another installation's file in `/etc` says nothing about this runtime). Only when `buildcfg` is unavailable, or prints no `SYSCONFDIR`, are the guesses used: beside the install prefix, `/usr/local/etc` and `/etc`. A `no`, a file that exists but cannot be read, and a value that is neither `yes` nor `no` are refusals, and a configuration that does not mention the setting is the runtime's default, `yes`. For `runtime: python` it also runs a preflight in the job's own image with the same isolation flags and checks that the container's PID namespace is not the launcher's; that proof stands even where no configuration is found. A user image (`runtime: image`) has no Python to rely on, so only a configuration that is found, readable and says `yes` verifies it; with none, the job is refused. The log names which check ran (the configuration file, or the preflight probe and what the configuration said).
- **Bind sources are checked.** Immediately before the container runs, every bind source is examined with `lstat` and must be a real directory, not a symlink; a path holding `,` or `:` is refused (the runtime splits a bind argument on them). `runtime_error`, detail `bind_source_unsafe`.
- The container's environment is the job's allowlisted variables and `NEMAR_OUT`, nothing from the host, plus `CUDA_VISIBLE_DEVICES` for a GPU job when the scheduler's value is plain (letters, digits and `,:_.-`, at most 128 characters).
  The runtime reads each `--env` argument as a CSV record, so a value with a comma or a double quote is passed quoted as CSV (`"NAME=a,b"`, the quote doubled); the tests run that against a real runtime.
- The runtime process itself gets only `PATH`, `HOME`, `USER`, `LOGNAME`, `LANG`, `LC_ALL`, `LC_CTYPE`, `TMPDIR` and `XDG_RUNTIME_DIR`, because `APPTAINERENV_*` and `SINGULARITY_BIND` style variables in the launcher's environment would otherwise be honoured.
- `launcher/` is a sibling of `work/` and is not mounted, so the container can neither read nor forge the result, the provenance or the manifest.
- `/code` and `/data/<id>` are read-only.
- The container is started in a session of its own, with `oom_score_adj` raised to 1000, so the kernel's OOM killer ends the entry rather than the launcher that has to record it.
  After a SIGTERM the container gets 10 seconds to stop, and the whole process group is then killed (a process that is PID 1 of its namespace ignores a SIGTERM it has no handler for); whatever the entry left running in the group is killed when the container exits.
- Exit status 255 is `entry_failed` like any other non-zero status, and is charged: the runtime's own fatal exit and an entry that exits 255 (or `-1`) cannot be told apart once user code could have run, and refunding it would let any entry refund itself. `runtime_unusable` is only for a runtime binary that cannot be started and for a preflight that failed, both before any user code.

The entry's standard output and error go to Tapis's `tapisjob.out`; the launcher's own log goes to `launcher/launcher.log` and to standard error.
`tapisjob.out` is user output: NEMAR must never parse it.

Before anything is downloaded, in this order: the runtime and the image (`no_container_runtime`, `image_missing`), the bundle and the entry (`entry_not_found`), the PID namespace check, then the datasets, so a missing entry wastes no download and a staging failure never masks `entry_not_found`.

## Bundle

`POST <api_base>/compute/jobs/<job_id>/bundle` with the token returns a ten-minute presigned address of `bundle.tar.gz`; it must be HTTPS (HTTP only on loopback).
The launcher downloads it to scratch with the same retry, size and SHA-256 checks as a dataset file, then inspects every member of the archive before extracting any.
A redirect is followed only to an allowed address, never from HTTPS down to HTTP.
An archive is refused whole when a member is absolute, contains `..` or an empty segment (`a//b`), is a link, a device, a FIFO or another special file, appears twice, or has a file as a parent.
`tar -czf bundle.tar.gz -C dir .` writes `.` and `./name`: a leading `./` is stripped and the root entry is allowed.
Reading is bounded: more than 100,000 members (counted while the headers are read), more than 4 GiB of file data, a decompressed stream past that plus 128 MiB, or a long-name or pax header over 1 MiB is `archive_too_large`.
Extraction is hand-written, creates regular files and directories only, and removes the setuid, setgid, sticky and group and world write bits.

## Datasets

For each dataset the launcher reads `GET <api_base>/data/<id>/<version>/manifest.json` (entries `{path, size, checksum?, checksum_algorithm?, bytes_url, url}`, at most 38,000), where `<version>` is the tag from the specification.
Each `bytes_url` must be HTTPS (HTTP only on loopback).
It selects the files the `include` globs name (all when there are none), and checks the selection against `limits` before fetching any byte.

**The `include` dialect** is the shared contract's (the `include` description in the schema), matched against manifest paths relative to the dataset root, case-sensitive, with no escapes:

- `**/` at the start of a path segment matches zero or more directories (`a/**/b` matches `a/b`, `**/README` a top-level `README`);
- `**` anywhere else matches any run of characters including `/` (`a**b` matches `a/x/b`, and `sub-01/**` everything inside `sub-01`); three or more stars in a row are this too, never `**/`;
- `*` matches any run of characters and `?` exactly one, and neither ever matches `/`;
- `[abc]`, `[a-c]` and `[!abc]` are classes, by code point; a class never matches `/`, so `[!a]` is any one character except `a` and `/`;
- a class ends at the first `]` after its start, so `[]` and `[!]` are empty classes; a range that ends before it starts (`[z-a]`) is refused too; a `[` with no `]` after it is a literal `[`;
- a caret is an ordinary character everywhere (`[^a]` holds a caret and an `a`), and a leading dot is not special.

An empty class and an inverted range are `bad_include`, and that is the whole of what is refused (`include_pattern_problem`, checked against nemar-cli's own lists by the drift tests).
Every pattern is compiled before any file is selected, so a refused pattern is `bad_include` whatever else the job names.

Matching never backtracks: a pattern is compiled to an automaton whose states are the bits of one integer, and a path costs its length times a few integer operations (`launcher/nemar_launch/include.py`).
A regular expression, or a glob that backs up to an earlier `*`, can take exponential or cubic time on a hostile pattern, and a job's author writes the pattern.

- **bind**: when `NEMAR_LOCAL_DATASETS_ROOT` holds `<id>/<version>/` (a real directory, not a symlink) and every selected file there matches the manifest by size and checksum (`SHA256E` sha256, or the git blob SHA-1 for `git`), that directory is mounted read-only; no file is requested.
  A symlink, a missing file, a different byte or an entry without a checksum falls back to download.
  The whole directory is mounted, so files the `include` globs did not select are visible too; only the selected files are verified.
- **download**: each file from its `bytes_url` into scratch, five attempts with backoff and jitter, `Range` resume after a truncated body, size and checksum checks, four files at a time.
  A 502 whose `error` is "Published file is missing from storage" is lost content: it is not retried and its path is logged.
  Any other 502 is a transport fault and is retried.

A manifest path that could leave the dataset directory is refused before anything is downloaded.
A SIGTERM during staging stops it: the entry does not run and the staging record does not claim files were staged.

## After the entry exits

The launcher reads `work` back with the container gone.
The walk is iterative (an explicit stack, so a 1,000-level tree cannot raise a recursion error) and changes directory into each directory in turn, so a path longer than the system's limit is no obstacle.
Every directory is first made owner-readable, writable and searchable (`u+rwx`), so what a read-only or mode-000 directory holds can be listed and removed.

- A symlink anywhere under `work` is unlinked and recorded in the log; so is a FIFO, socket or device (a hash of a FIFO would never return). One that cannot be unlinked is recorded and left out: it never reaches the manifest or the upload.
- Only regular files under `work/outputs` enter the manifest, sorted by path, hashed through a descriptor opened without following a link whose size must still match.
- A name the Worker would refuse is left out and recorded: a backslash or control character, an empty, `.` or `..` segment, more than 512 UTF-16 code units, or a results key `jobs/<id>/outputs/<path>` of more than 1,024 UTF-8 bytes.
- **Limits are enforced before any file is hashed.** More than 100,000 outputs is `too_many_outputs`, and so are outputs whose entries alone would make the outputs request larger than the Worker's 1 MiB cap (counted as the files are met, so none is hashed for a request that cannot be sent); a file over 5,000,000,000 bytes is `results_too_large`; there is no limit on the total, because `limits` bound the dataset selection only and the Worker's route refuses an excess (charged as the same word). Each ends as `result_upload_failed` with that word as the detail (the submitter's, not the platform's), the manifest is empty and no output is uploaded.
- **An output that cannot be read** (a mode-000 file, or a directory that cannot be listed even after the `chmod`) is never skipped silently: the job ends `runtime_error` with detail `outputs_unreadable`, the manifest is empty and no output is uploaded. A mode-000 directory is made searchable and read normally.

`provenance.json` holds the launch specification digest, the bundle digest, the image URI and digest, a free-text runtime version (the container runtime's own version line and the host Python), the staging records (`id`, `version`, `mode`, `files`, `bytes`), phase timings (`code_fetch`, `staging`, `run`, `finalize`), the Slurm job id and the entry's exit code.
`result.json` holds the failure code, a fixed-word detail, the exit number and the entry's exit code.
All three are validated against the schema's `$defs` in the tests.

## What is guaranteed

Once the specification parsed, however a phase ends, **including an unexpected error in the launcher itself** (a bug, a fault nothing planned for):

- `result.json`, `provenance.json` and `manifest.json` are produced, and `result.json` names the FIRST failure, or none. They are kept in memory as well as written into `launcher/`; if the directory cannot be written they are sent from memory.
- the four launcher files are uploaded: by the full upload when it can complete, and otherwise by the launcher-only upload (no `complete`).
- the exit number is 0, or 10 to 16 for the first failure.
- an unexpected error is `runtime_error` (or `result_upload_failed` during the upload) with detail `internal_error`, and only the exception's class name is logged.

Not guaranteed: a specification that does not parse has no API address, so nothing is uploaded (the three files are still written); a refused output directory gets no file written into it; a launcher killed with SIGKILL (the kill signal, which no program can catch), or a node that dies, leaves what it had written and nothing else.
A SIGTERM that arrives after the container exited is still recorded: the record never says success for a job Slurm terminated, and once the stop event is set no further output upload starts.

## Upload

After the documents are written the launcher uploads, with the token only on NEMAR's own routes:

1. `POST <api_base>/compute/jobs/<job_id>/outputs`: every output's path, size and SHA-256, and the sizes of the four launcher files; the answer is one presigned address per file. The request is validated against the schema's `outputs_request` and capped at 1 MiB (the Worker's own cap) before it is sent; over the cap is `too_many_outputs`. Each address must be HTTPS (HTTP only on loopback).
2. A PUT of each file to its address: outputs first, four at a time; then the launcher files in `codes.UPLOAD_ORDER`, `result.json` last. Three attempts each with backoff; no redirect is followed; the token is not sent. The launcher files are sent from memory (the documents, and a snapshot of the log as it was when its size was declared, although the launcher keeps logging).
   An address lives about an hour and a large upload can outlive it, so an expired address (403) is **re-requested up to 2 times for each file, and at most 50 times for the whole run**, then the file fails. The launcher-only follow-up has a budget of its own.
3. `POST <api_base>/compute/jobs/<job_id>/outputs/complete` after every PUT succeeded. `complete` is the commit: files without it are not a result. It is retried on a transport failure and, three times in all, on a 409 `results_incomplete`, so **NEMAR may see `complete` more than once and must treat it as idempotent**.

A job that failed earlier still uploads what it made and the launcher files, and calls `complete`.

When the full upload fails (a refusal for a size cap or a quota, a PUT that fails past its retries, a refused `complete`), the launcher makes one more request that names no outputs and uploads only the four launcher files, so NEMAR can read why; `complete` is not called then.
The failure is `result_upload_failed`, and it never replaces an earlier failure code.

No per-object checksum header (`x-amz-checksum-sha256`) is sent on a PUT: it has to be signed by the Worker with the address, and is decided with a real R2 check in the gateway epic (see [unverified.md](unverified.md)).

## Failure codes, exit numbers and details

Exactly these seven codes, in nemar-cli's order; the first failure wins.

| Exit | Code | When |
|---|---|---|
| 0 | none | nothing failed |
| 10 | `code_fetch_failed` | the bundle route refused, the download failed, the digest or size differs, the archive is unsafe |
| 11 | `dataset_fetch_failed` | a manifest, file, checksum, size, path or cap failure while staging |
| 12 | `entry_not_found` | the entry is not a file in the bundle |
| 13 | `runtime_error` | the entry exited non-zero; an unsupported runtime or invalid specification; the output directory, bind sources, PID namespace, runtime or image refused; an output that cannot be read; a SIGTERM before the time limit |
| 14 | `out_of_memory` | exit status 137 (SIGKILL) |
| 15 | `time_limit` | a SIGTERM at or after `SLURM_JOB_END_TIME` minus 120 seconds |
| 16 | `result_upload_failed` | the upload failed or the results were over a limit, and nothing failed earlier |

**Every detail the launcher can emit is a declared word**, listed per code in `codes.LAUNCHER_DETAILS` and shared with nemar-cli (`LAUNCHER_DETAILS` in `shared/compute-state.ts`), so NEMAR decides who pays from a table:

| Code | Details |
|---|---|
| `code_fetch_failed` | `unsafe_archive`, `archive_too_large`, `bad_response`, `fetch_failed`, `checksum_mismatch`, `digest_mismatch`, `size_mismatch`, `content_missing`, `disk_error` |
| `dataset_fetch_failed` | `bad_include`, `too_many_files`, `too_many_bytes`, `dataset_not_found`, `bad_manifest`, `manifest_too_large`, `unsafe_path`, `bad_response`, `fetch_failed`, `checksum_mismatch`, `size_mismatch`, `content_missing`, `file_not_found`, `disk_error` |
| `runtime_error` | `entry_failed`, `terminated`, `outputs_unreadable`, `invalid_spec`, `unsupported_runtime`, `scratch_in_output`, `scratch_unavailable`, `no_container_runtime`, `image_missing`, `runtime_unusable`, `internal_error`, `output_dir_not_empty`, `bind_source_unsafe`, `pid_namespace_unavailable` |
| `result_upload_failed` | `results_too_large`, `results_quota_exceeded`, `too_many_outputs`, `outputs_unreadable`, `bad_response`, `fetch_failed`, `incomplete_plan`, `upload_failed`, `outputs_refused`, `complete_refused`, `internal_error` |
| `entry_not_found`, `out_of_memory`, `time_limit` | none |

A transport fault (a status, a timeout, a dropped connection, a refused redirect) is `fetch_failed`, and a Worker route's own refusal is `outputs_refused` or `complete_refused`; the exact word (`http_503`, `timeout`, the route's `code`) is kept in the log line only.
The three results refusals `results_too_large`, `results_quota_exceeded` and `too_many_outputs` are recorded as the route named them.

Who pays: a detail is the platform's, and the job is refunded, if and only if a submitter cannot cause it with a valid specification and bundle (`codes.PLATFORM_LAUNCHER_DETAILS`): everything above except `unsafe_archive`, `archive_too_large`, `bad_include`, `too_many_files`, `too_many_bytes`, `entry_failed`, `terminated`, `outputs_unreadable`, `results_too_large`, `results_quota_exceeded` and `too_many_outputs`, which are the submitter's and are charged, even under a fetch code.
A signal death of the entry is reported as 128 plus the signal, as a shell does, in `entry_exit_code`.

## Environment

| Variable | Who sets it | Meaning |
|---|---|---|
| `NEMAR_LAUNCH_SPEC` | the Worker, through Tapis | the launch specification as base64url of its JSON text; holds the job token |
| `_tapisExecSystemOutputDir` | Tapis | the output directory |
| `NEMAR_HOST_PYTHON` | the application definition, optional | the host Python to use |
| `NEMAR_LOCAL_DATASETS_ROOT` | the application definition, optional | root of NEMAR's own copy of datasets, `<root>/<id>/<version>/` |
| `NEMAR_SIF_ROOT` | the application definition | the directory `image.sif` is looked up in; required |
| `SLURM_TMPDIR`, `TMPDIR` | Slurm, the system | where scratch goes, in that order |
| `SLURM_JOB_END_TIME`, `SLURM_JOB_ID` | Slurm | the time limit rule, and the job id in provenance |
| `CUDA_VISIBLE_DEVICES` | Slurm | passed into the container for a GPU job when plain |

Tapis archives `tapisjob.env` by default and the specification carries the job token, so the launcher never prints the specification, the token or a presigned address.
Logs show a URL as host and path only (`net.redact`, which never raises), and never copy a response body.

## Developing

```bash
uv sync --frozen
uv run pre-commit install        # once per clone: ruff on the staged files, with --fix --unsafe-fixes
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest --cov=launcher --cov-fail-under=85
uv run --no-project --python 3.9 --with pytest --with jsonschema pytest tests/launcher
uv run python scripts/build_launcher_zip.py --check
```

The tests use real files, real processes and local HTTP stand-ins built on the standard library (`tests/standin/`): a data plane over `tests/fixtures/dataset/` that, like the real one, serves only `latest` and `vN.N.N`, NEMAR's compute routes with an object-storage PUT receiver, and a strict stand-in for the container runtime binary for the tests that are about the launcher's flow.
No response in a stand-in was recorded from the real service; the shapes come from the contract files.
The wrapper tests run the zip unpacked the way Tapis does.

The container tests (marker `apptainer`) need a real Apptainer or Singularity binary.
They skip locally, with the reason stated, where there is none, and run in CI against a pinned Apptainer and a small public image pinned by digest.
In CI `NEMAR_REQUIRE_APPTAINER=1` turns a missing binary into a failure.
The runtime configuration tests (`test_runtime_config.py`) also change the runtime's system configuration with `sudo`, as the hosted CI runner permits, and put it back.
To run the container tests on a Linux machine with Apptainer, `uv run pytest -m apptainer`; `NEMAR_TEST_SIF` names a prebuilt image to skip the pull.
