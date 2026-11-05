# What is not verified yet

The launcher is tested against local stand-ins and, in continuous integration (CI), against a real Apptainer 1.5.5 on a hosted Linux runner.
It has not run on a compute node.
These are the facts it depends on that nobody has confirmed on the node or against NEMAR's Worker, in the order they would bite.

## What CI has observed, and what it is not

Observed on the hosted runner with Apptainer 1.5.5 (the community build), and pinned by tests that run there:

- An entry killed by SIGKILL (the kill signal, the one the kernel's out-of-memory killer sends) is reported with exit status 137, and the launcher records `out_of_memory` with `entry_exit_code` 137.
  Any other signal death is reported as 128 plus the signal (SIGSEGV, 11, is 139) and is `runtime_error`.
- A SIGTERM (the termination signal) sent to the container's process group stops `sleep 60` at once, run directly or through a shell, and the runtime reports exit status 143 (128 plus 15); so the job ends `time_limit` or `terminated` as the end time dictates, with the launcher files uploaded.
  The container's process group is killed after 10 seconds if the entry has not stopped (an entry that ignores SIGTERM is the case for that; the stand-in runtime tests it).
- A container started with `--containall` has a process identifier (PID) namespace of its own, a private `/dev/shm`, no view of the host's home directory, and `oom_score_adj` 1000 as the launcher set it.
- A `bind path` line in the runtime's system configuration mounts a host directory into every container unless the container is contained (`--contain`, so `--containall`) or `--no-mount bind-paths` is given; the control runs show each case (see `tests/launcher/test_runtime_config.py`).
  The launcher gives both, so on this runtime either alone would do; the node's runtime may differ.
- A configuration with `allow pid ns = no` is refused by the launcher before any user code runs.

None of this has been observed on the node's runtime, which may be a commercial Singularity.

## The node

- **Host Python.** The launcher needs Python 3.9 or newer on the node, outside the container.
  If the node has only an older `python3`, the wrapper exits 13 with a message and the application definition has to name an interpreter (`NEMAR_HOST_PYTHON`).
  This is an escalation to the owner, not something to work around in the launcher.
- **Outbound Hypertext Transfer Protocol Secure (HTTPS) from compute nodes** to NEMAR's application programming interface (API) and to the object storage endpoint the presigned addresses point at.
  It was measured as available in October 2026; it is still a question for the center that runs the nodes.
- **The container runtime's behaviour on the node.** CI uses the community Apptainer; the node may run a commercial Singularity.
  `--containall`, `--cleanenv`, `--no-home`, `--no-mount bind-paths`, `--env`, `--bind ...:ro` and `--nv` are common to both as far as the documentation says, but the isolation tests have not run there.
- **Site bind paths.** `--no-mount bind-paths` turns off the `bind path` lines of the runtime's configuration; whether the node's runtime honours the flag, and what else its configuration mounts (`mount hostfs`, `mount tmp`), is unconfirmed. Check the site's `apptainer.conf` or `singularity.conf`.
- **PID namespace verification.** The launcher fails closed: with no way to verify the namespace it refuses before any user code (`pid_namespace_unavailable`). It reads `allow pid ns` from the runtime's configuration: the `SYSCONFDIR` that `<runtime> buildcfg` prints is authoritative (only `<SYSCONFDIR>/<runtime>/<runtime>.conf` is read, and a missing file there is "cannot be verified"), and only when `buildcfg` is unavailable or prints none are the usual places tried (beside the install prefix, `/usr/local/etc`, `/etc`). For `runtime: python` a preflight probe in the job's image also runs, and proves the namespace even where no configuration is found.
  For `runtime: image` (a user's image, no Python to count on) only the configuration can verify it, so a user image on a node whose configuration cannot be located or read is refused and not run.
  Whether `buildcfg` prints `SYSCONFDIR` on the node's runtime, and whether the file is where it says, is unconfirmed; CI runs against the real Apptainer 1.5.5 and requires its configuration to be found and read that way.
- **How `--env` is parsed.** Apptainer 1.5.5 reads the flag as a comma-separated values (CSV) record, and the launcher quotes values accordingly; whether the commercial Singularity on the node parses it the same way is unconfirmed (a value with a comma is the case to try).
- **The network namespace is shared.** `--containall` does not isolate the network: the entry can reach whatever the node can, including NEMAR's API.
  The container has no job token (its environment is the allowlist), so it cannot call the job routes, but it is not cut off from them or from the node's network.
  Whether `--net --network none` is permitted on the site is a question for the site operator, an owner decision; nothing in this phase uses it.
- **Manifest addresses.** A manifest entry's `bytes_url` may name any HTTPS host (or loopback HTTP); the launcher trusts NEMAR's data plane to name its own, and moving to an allow-list of hosts was declined.
- **Graphics processing unit (GPU) visibility.** `CUDA_VISIBLE_DEVICES` is passed into the container for a GPU job when its value is plain; whether the scheduler sets it, and whether the device control group alone shows the container the right devices, is unconfirmed.
- **Node-local scratch is shared.** The container's `/tmp` is bound to node-local scratch (not to the archived directory), so one job can fill a shared node's scratch and co-located jobs then fail with `disk_error` or `scratch_unavailable` (refunded as the platform's). The launcher removes its own scratch tree afterwards, whatever modes the entry gave the directories in it, but it sets no quota: a per-job quota or a scratch size limit is a question for the site operator.
- **The image file.** `image.sif` is `<64 hex digits of the image digest>.sif` under `NEMAR_SIF_ROOT`, which the application definition sets; there is no fallback.
  A Singularity Image Format (SIF) file's content hash is NOT verified: only its name ties it to the digest, so whoever can write the root can substitute an image.
- **Slurm's environment.** The launcher reads `SLURM_JOB_END_TIME` (the time limit rule), `SLURM_JOB_ID` and `SLURM_TMPDIR`.
  Whether a Tapis ZIP-runtime job sees them, and how long Slurm waits between SIGTERM and SIGKILL (the record must be written inside it; the launcher gives the container 10 seconds), is unconfirmed.
- **Tapis.** The ZIP runtime runs `./tapisjob_app.sh` and reads `tapisjob.exitcode`; the launcher writes that file itself and exits with the same number.
  Whether the runtime keeps the script's mode bits, and what it does with a non-zero exit, is unconfirmed.
  `tapisjob.out` holds the entry's output and is untrusted: NEMAR must never parse it for a result.

## NEMAR's data and storage

- **The local dataset root layout and its protection.** The launcher binds `<NEMAR_LOCAL_DATASETS_ROOT>/<id>/<version>/` (a real directory, with `<version>` the data-plane tag: `latest` or `vN.N.N`) when every selected file there matches the manifest.
  Whether NEMAR's copy on the node is laid out that way, and holds plain files and not symlinks into an annex object store, is unconfirmed; a symlink never matches, so a git-annex checkout would always fall back to download.
  The root must be immutable and not writable by the shared account that runs the jobs: bind mode verifies the selected files by checksum and then mounts the whole `<id>/<version>` directory read-only, so the unselected files in it are visible to the job and are not verified (documented; an owner decision).
- **The spelling of the checksum algorithm in a live manifest.**
  The manifest of the mirrored dataset `on007823` at `v1.0.0` (read on 2026-11-04) spells it lowercase `sha256` for its 170 annexed files and `git` for the other 1,363, while the shared contract names annex backend names such as `SHA256E`.
  `net.Checker` knows `SHA256`, `SHA256E`, `SHA512`, `SHA512E` and `git`, so an entry spelled `sha256` is checked by size only and is never bound from a local copy.
  Unconfirmed: which spelling the Worker emits for other datasets, and whether the launcher should accept the lowercase one; the fixture's committed `manifest.json` and the stand-in data plane use `SHA256E`, as the contract and the stand-in do.
- **Dataset version forms.** The data plane accepts only `latest` and `vN.N.N`. The launch specification carries that tag and the launcher uses it verbatim; the Worker resolves the user's spelling. The stand-in data plane enforces the same two forms; the real one has not been called by the launcher.
- **Presigned multipart upload to the results bucket.** A single PUT per file is the only upload the launcher does, so a single result file is limited to 5 GB until multipart upload through presigned addresses is verified on Cloudflare R2 object storage.
- **What the presigned addresses require.** The launcher PUTs with `Content-Type: application/octet-stream`, a `Content-Length`, and no token; if the Worker signs other headers, the PUT fails with a 403 and the address is re-requested (up to 2 times for each file).
  It sends a `nemar-launch/<version>` `User-Agent` on every request, in case a content delivery network rejects the default.
- **A per-object checksum header (deferred).** A `x-amz-checksum-sha256` header on each output PUT would let storage verify the bytes, but it has to be signed by the Worker together with the address, so the launcher does NOT send it. Decide it with a real R2 check in the gateway epic.
- **The Worker's responses.** The outputs and bundle routes do not exist yet; the stand-ins follow the schema's `$defs`.
  The error body is read for a `code` word only, and only the three results refusals (`results_too_large`, `results_quota_exceeded`, `too_many_outputs`) are recorded as they are; any other refusal is `outputs_refused` or `complete_refused`.
  The launcher posts `{}` as the body of the bundle and complete routes, retries `complete` on a 409 `results_incomplete`, and so may call it more than once: it must be idempotent.
- **Who pays for a detail.** `codes.PLATFORM_LAUNCHER_DETAILS` says which declared details are the platform's (refunded) and which the submitter's (charged); a transport fault is `fetch_failed`, the platform's.
  A status of 255 after user code started is `entry_failed` and charged, because the runtime's own fatal exit cannot be told from an entry's: a platform fault that surfaces as 255 mid-run is therefore charged to the submitter (a known limitation, and a question for the Epic A refund path).

## Not done in this phase

- No image is published by the launcher's phase; the container tests use a small public image. The runtime image, its build and its smoke tests are in [images.md](images.md).
- The Tapis application and system definitions and the observer are written and validated in CI (see the last section), but nothing is registered with a tenant and nothing has run on a node.
- A symlink or special file that cannot be unlinked even after the directory's permissions are widened, a directory that cannot be listed even after the same, and an entry that vanishes between the directory listing and its `lstat`, have no test: an unprivileged test cannot make any of them on demand (the walk records each as `outputs_unreadable` or skips it, and the code paths are read, not run).

## The runtime image (nemar-python)

[images.md](images.md) says what the image is and how a consumer refers to it.
CI builds the image, builds a SIF from it and smoke-tests both, with the community Apptainer 1.5.5.
These are the facts that nothing has confirmed yet.

- **The push, sign and release path has not run.** Pushing to GHCR, `cosign sign`, `cosign verify`, the SIF built from a digest, its push as an OCI artifact and the release asset all run only on a version tag or a dispatch from `main`, and only for a commit reachable from `main`.
- **SIF compatibility with the cluster's Apptainer variant is unverified.** The cluster may run a commercial Singularity; a SIF from this image has not been run there. It is the operator's test, with `--containall --cleanenv --no-home --no-mount bind-paths` and the smoke test.
- **The image's environment under `--cleanenv`: observed once, with a limited scope.** The Dockerfile sets `PATH` (the virtual environment first), the thread counts and `PYTHONHASHSEED`; the launcher runs the entry with `python` from the image's `PATH`, so the launcher depends on them surviving `apptainer exec --containall --cleanenv`. The `test` job of run 41909394410 (pull request 8) passed that check: the community Apptainer 1.5.5 on a hosted runner, with a SIF built from a local image archive, not from the pushed digest, and not the cluster's variant.
- **A SIF's content hash is not verified by the launcher** (see above); the release's digest file records the file's SHA-256, but it is an unsigned release asset, so it guards only against a corrupted copy.
- **The existence checks rely on GHCR answering with exactly 404.** `scripts/check_tag_absent.py` lets a publication go on only when a manifest `HEAD` for the tag answers 404, and stops on every other answer. Unconfirmed: that GHCR answers 404 for an absent tag, including in a package that does not exist yet, and that its token endpoint accepts the actor and the workflow's token as basic authentication. If the first release stops at this check, the check needs a change; it is not bypassed.
- **Signature verification** needs cosign 3 or newer; older versions may not read signatures made by cosign 3. What a passing check proves, and what it does not, is in [images.md](images.md).
- **Build requirements are pinned by version, not by hash.** `docopt` and the signalJourney validator are built from source during the image build, in an isolated environment; their build backend (`setuptools`) is pinned to the locked version, but PyPI is still trusted for the bytes of that version.
- **Thread counts.** The image pins one thread per math library; a recipe that wants more must set them, and what is right on a node is unmeasured.

### Owner items for the image

None of these is in place; each needs a repository setting, not a code change.

- **Protect publication.** The workflow refuses a commit that is not reachable from `main`, but that check lives in a file the tagged commit supplies. A tag ruleset on `image-nemar-python-v*`, branch protection or a ruleset on `main`, or an environment with required reviewers on the `publish` and `sign` jobs (which the workflow does not name today) would stop a person with write access who edits the file and pushes a tag. Until one exists, a signature that passes the documented check does not mean the commit was reviewed.
- **GHCR.** Confirm that publishing is enabled for the repository and set `nemar-python` and `nemar-python-sif` public so a node can pull without credentials.
- **The SIF package.** `apptainer push` creates `nemar-python-sif`, and it cannot set `org.opencontainers.image.source`, so GHCR does not link the package to this repository by itself. The first release creates it with the workflow's token; link the package to the repository and give the workflow write access before the first release. The second release is the likely point of failure; [images.md](images.md) says what to do when a run fails after the image was pushed (re-run only the failed jobs, never use an unsigned tag, publish a new version if the SIF tag already exists).

## The recipes (`nemar-floor`)

[recipes.md](recipes.md) says what the recipe does.
These are the facts nothing has confirmed yet.

- **Equality across hosts.** Everything was run on one machine (macOS arm64, NumPy on Apple's Accelerate). Two runs there are byte-identical, but a borderline channel or window decision could flip on Linux, where NumPy uses OpenBLAS. `tests/recipes/test_golden_decisions.py` pins the fixture's decisions from the reference machine, and the first Linux CI run is what settles it: a difference there is a finding to record, not to hide.
- **The thread pin on Linux.** `threadpoolctl` detects no pool on the reference machine, so the recorded inventory is empty there and the test that every pool reports one thread skips with a stated reason; CI on Linux has the pools and runs it. The pin itself (the thread variables set before a library loads, `threadpool_limits` for the pools already loaded) is observed locally through torch's OpenMP pool only.
- **The launch end to end in the SIF.** There is no container runtime on the development machine. The test runs with the stand-in runtime locally and in CI (`ci.yml`), and in the real SIF in the image workflow's pull-request job, where it has not yet been seen to pass until that job runs. Cross-environment equality of the state files between the container and the host is asserted there for the first time.
- **Units.** `robustAmplitude` is in the unit the loader delivers: uV for EDF and BDF, as read. A source in another unit would be wrong by a constant factor; it is recorded as "as loaded" and not verified for other sources.
- **Loaders.** The BrainVision route was run by the phase plan on a real recording, and `.edf` and a set saved with an empty `etc` here. A `.set` with a separate `.fdt`, a MATLAB v7.3 `.set`, and `.bdf` from the fixture's own acquisition system were read in eegprep's source and not run.
- **A recording whose every channel is flat.** eegprep's flatline step keeps every channel when all are flat (observed on a constant set), so the later steps run on them and the floor says nothing special; not handled and not tested.
- **A truncated EDF** loads with fewer samples and only a reader warning. The launcher's size and checksum checks, and the SHA-256 the recipe records for each input, are what catch a damaged copy; the recipe adds no check of its own.
- **Memory.** A run holds about three float64 copies of the longest recording at its peak. For a recording of an hour at 1 kHz with 128 channels that is about 11 GB; nothing was measured at that size. A `MemoryError` is a recorded failure and a SIGKILL by the kernel is the launcher's `out_of_memory`.
- **Staging cost.** pybids indexes the whole staged tree once per process; the cost for a job that stages tens of thousands of files is unmeasured. Job authors include `dataset_description.json` and the recording's `channels.tsv`, `electrodes.tsv`, `coordsystem.json` and `eeg.json` and nothing else they can avoid.
- **The dataset record's size cap** truncates the embedded rows above roughly 1,500 recordings (about 130 bytes a row against a 200,000-byte budget); the table file is complete and hashed.

## The Tapis definitions and the observer

[operator.md](operator.md) says what the definitions are and how they are registered.
Nothing here has run against a real tenant or a real node.
The application and system schemas are derived from Tapis's own OpenAPI files (commits recorded in `deploy/tapis/openapi/SOURCE`), so a field name is as Tapis spells it, but the specifications say nothing about behaviour, and no stand-in response was recorded from a real tenant.
Answers marked "from source" were read in the Tapis source (branch `local`, release 26Q2.1) by a reviewer; each is to be confirmed on the tenant, and each says so.
These are the assumptions, in the order they would bite.

### The encoding of the launch specification, and what still blocks registration

**Registration is BLOCKED.**
Nothing is to be registered with any tenant until the owner items recorded on nemarOrg/nemar-process#9 are settled:
the release-asset procedure (immutable releases, tag rulesets and a reviewed release workflow),
whether the entry scripts delete the launch files after the job starts,
whether the launcher adds `--no-mount hostfs`,
and confirmation on the tenant of every Tapis behaviour marked "from source, to confirm on the tenant".
The encoding of the launch specification is no longer one of them:
the value of `NEMAR_LAUNCH_SPEC` is the base64url encoding (no padding) of the specification's JSON text,
so a shell that sources the launch file reads only letters, digits, `-` and `_`,
and the launcher decodes it strictly.
No job with a real token is to be submitted before then either, and the observer's application waits with the rest.
A dry run of `scripts/register.py` is safe.

Tapis writes each environment variable into launch files (`tapisjob.env` and `tapisjob.sh`) that a shell reads, and the launch specification travels in one of them.
Its first design sent the JSON text, which is not safe there: the scheduler quotes a value only conditionally, so a value with no space loses its quotes, and a `$`, a backtick or a backslash in a user's string is expanded or breaks the quoting.
The value is now the text in base64url without padding ([launcher.md](launcher.md)), made only of `A-Za-z0-9-_`, which no shell treats as special whether or not the scheduler quotes it.
This repository's tests show that against a stand-in of the quoting rule, with real shells; they do not show what Tapis does.
Still unconfirmed, each to be confirmed on the tenant:

- **The quoting rule.**
  From source, to confirm on the tenant: the scheduler leaves a value bare unless it holds a space, one of `&`, `>`, `<`, `|` or `;`, or a backtick, and otherwise double-quotes it and escapes only the double quote.
  The encoded value is safe under that rule and under any other that leaves letters, digits, `-` and `_` alone.
  After the first job, read the launch file in the execution directory and compare its line with the value NEMAR sent.
- **Any other change to a value.**
  Whether Tapis trims, wraps, expands a macro in or limits the length of a value is unconfirmed (the longest value is 87,382 characters).
  A changed value is refused as `runtime_error` / `invalid_spec`.
  When a wire rule fires, the log names it (`empty`, `too_long`, `not_base64url`, `bad_length` or `not_printable`) and no part of the value.
  A value cut at a group boundary decodes to a truncated text and is refused by the JSON rules with no word, so a bare `failed: runtime_error (invalid_spec)` on a large job also points at truncation.
  A macro such as `${JobUUID}` cannot match, since the value holds no `$` and no brace.

### Archiving and directories

- **The request is the control and the application is the fallback.**
  From source: a job request's own `archiveFilter.includeLaunchFiles` wins whenever the request sets it, and the application's value is used only when the request leaves it out; Tapis's own default is `true` (`JobParmSetMarshaller.mergeArchiveFilters`).
  NEMAR's client sets it to `false` on every job it submits, so the client is what keeps `tapisjob.env` out of the archive, and the applications' `false` covers a request that omits it.
  The setting is sometimes written `archiveIncludeLaunchFiles`, but the specification has no such field: the derived schema refuses an unknown key, and `archiveFilter.includeLaunchFiles` is the only field that exists.
  To confirm on the tenant: after the first job, that the archive holds neither launch file.
- **The applications' `excludes` do not reach the launch files.**
  From source: Tapis copies the launch files by name from the execution directory, and when the archive directory is the output directory it skips the filters entirely.
  The applications still list both files in `excludes`, as a record of intent and for the case the filter is applied; it is not the control.
- **The macros.**
  `${JobWorkingDir}` and `${JobUUID}` are Tapis's own and are not in the specification; their spelling and expansion are from the documentation.
- **Where Tapis creates a job's directories, and what it tells the job.**
  From source: Tapis creates the directories at `rootDir` plus the path, but the environment variables it gives the job (`_tapisExecSystemOutputDir` and the others) hold the rootDir-RELATIVE path.
  The launcher reads `_tapisExecSystemOutputDir` as a host path, so the two agree only when `rootDir` is `/`.
  The system template therefore sets `rootDir` to `/` and `jobWorkingDir` to the absolute `<scratch root>/jobs`, and `scripts/tapis_policy.py` computes the variable the way Tapis does and refuses to register a system whose root is not `/`.
  What keeps other accounts out of the scratch root is the host's own permission (mode 0700), not Tapis's `rootDir`.
  With `rootDir` `/`, any Tapis Files permission on `nemar-exec` reaches the community account's whole filesystem view, not only the scratch root; that is why the system is never shared with anyone but NEMAR's own service identity.
  If the first job on the `debug` partition shows the output directory somewhere else, stop: the launcher's directory contract is wrong and that is a launcher change, not a definition change.
- **Default archive location.**
  With no archive system fields, Tapis archives to its default place on the execution system.
  Where that is, and that it is not under a home directory, is unconfirmed.
- **What a read of a registered definition looks like.**
  `scripts/register.py` compares the stored definition with the rendered one, and also refuses a stored one that holds a forbidden field the rendered one leaves out (a non-null `archiveSystemId` or `archiveSystemDir`, a non-empty `archiveFilter.includes`, `fileInputs` or `jobEnvVariables`, and so on).
  Unconfirmed: that Tapis returns what was posted in the same order (a list it reorders is reported as a difference), and that it leaves `archiveSystemId` and `archiveSystemDir` null when they were not set (if it fills them with the execution system's defaults on read, every registered application will be reported as drifted; both are fail-closed noise, to be loosened only after a real read has been seen).
- **Job directories are never removed by Tapis**, so they accumulate (the observer adds one every 15 minutes); the site's scratch purge policy is what bounds them.
- **Tokens stay on disk.**
  Every launcher job's specification, and with it the job's token, stays in `exec/<job>/tapisjob.env` until the directory is purged, and the token is valid for the job's run (up to 7 days on the preemptible partitions).
  Deleting `./tapisjob.env` from the entry scripts after the job starts is an owner decision, pending a check on the tenant that nothing else reads the file; it is not implemented.

### Variables and the job request

- **A request can add any variable the application does not declare.**
  From source: only names starting with `_tapis` are reserved, and `FIXED` blocks only names the application declares.
  So `SLURM_TMPDIR` (which the launcher prefers to the fixed `TMPDIR`), `PATH`, `LD_PRELOAD` and the rest are guarded only by NEMAR's client, whose request schema declares exactly one variable, `NEMAR_LAUNCH_SPEC`, in a closed one-element array; `tests/deploy/test_tapis_client_agreement.py` pins that shape against the client's source.
  The definitions can add one guard of their own, and do: `NEMAR_HOST_PYTHON` is declared `FIXED` with an empty value, which (from source) Tapis accepts and which makes it refuse a request that sets a different value; the entry scripts read an empty value as "search the host".
- **The longest environment value.**
  A launch specification may be up to 64 KiB of JSON text, which is up to 87,382 characters as the base64url value.
  Tapis may limit an environment variable's length, or the length of the generated launch script.
  If it does, large specifications fail at submit; this is an escalation.
- **Empty `REQUIRED` variables.**
  The definitions give a `REQUIRED` variable no value; whether Tapis accepts that in a definition is unconfirmed.
- **`locked` at creation.**
  Both applications are created `locked`, so Tapis refuses to modify a version; whether Tapis accepts the flag on a create, and whether disabling a locked version still works, is unconfirmed.
  The owner can unlock a version (by Tapis's own route, with the owner's own tooling; `register.py` never calls it) to correct a mistake before first use.
- **The zip is fetched with no digest check.**
  From source: Tapis's zip stager fetches `containerImage` and unzips it, and checks nothing about its content.
  The URL is a GitHub release asset, which is mutable while immutable releases are off, so the digest in the definition's `notes` is only a record: `scripts/register.py --apply` downloads the file and refuses to register unless its SHA-256 is that digest, a ledger (`deploy/tapis/released-versions.json`) makes a released version's digest permanent, and the publishing steps are the owner's ([operator.md](operator.md)).
  Whether the tenant fetches a URL that redirects to object storage, and whether it restricts which hosts a ZIP application may come from, is unconfirmed.
  No workflow publishes the zips as release assets yet.

### Running a job

- **Reading `tapisjob.exitcode`.**
  From source: for a `FORK` ZIP job Tapis checks the process and then reads `<output directory>/tapisjob.exitcode`, and a missing file counts as success; the observer's entry script therefore writes it, as the launcher's does.
  For a `BATCH` job Tapis documents the file for `FORK` jobs only; how it turns a non-zero exit of a `BATCH` ZIP job into `FAILED` or `FINISHED` is unconfirmed.
- **Slurm's grace period.**
  How long Slurm waits between SIGTERM (the termination signal) and SIGKILL (the kill signal) when a job is cancelled or reaches its limit, its `KillWait` setting, is unconfirmed.
  The launcher records the result inside 10 seconds of the signal; a shorter wait loses the record.
- **The account option.**
  The system gives each logical queue a scheduler option `--account=<account>`, the CPU account for the CPU partitions and the GPU account for the GPU ones.
  Whether Tapis applies a logical queue's scheduler options (and not only a job's or an application's) is unconfirmed; so is the spelling of NEMAR's client's `--gpus=N`, listed in nemar-cli.
- **Memory.**
  From source: Tapis always sends `--mem`.
  A whole-node job that omits memory would be charged a full node but limited to the application's default, 2048 MB, so NEMAR's client must always send memory per partition (a note for phase A5).
- **Queue limits.**
  The logical queues' limits (nodes, cores, memory, minutes and the running-plus-queued job counts used as `maxJobsPerUser`) were read from the public Expanse user guide on 2026-11-04 and are unverified until measured; the template says so in its `notes`.
  The client sets no per-job ceiling for `large-shared` and the two preemptible partitions, so these limits are their only cap.
  The user guide gives some of its limits per group, not per user, and the preemptible partitions are not refunded; a limit that is too low only holds a job in Tapis, and one that is too high is refused by the scheduler.
- **No scheduler profile.**
  The system sets no `batchSchedulerProfile`, so no module is loaded for a job; whether the site needs one is unconfirmed.
- **Outbound HTTPS from the login node**, for the observer's post to NEMAR, is a separate question from the one for compute nodes; both are for the site operator.
- **The container's mounts (a launcher follow-up).**
  `--no-mount bind-paths` does not cover a site whose runtime configuration sets `mount hostfs = yes`, which would put the host's file systems, including the execution directories that hold tokens, in the container.
  Adding `--no-mount hostfs` is a change to the launcher's container arguments, which is a security change in another phase: the owner and driver decide.

### The observer

- **A `FORK` application on the login node (design question 6).**
  Whether the site allows `squeue`, `sinfo`, `sacct` and `expanse-client` to run on a login node every 15 minutes through Tapis is a question for the site operator, and the observer cannot run as designed without it.
- **The environment of a `FORK` job.**
  Whether it has the Slurm commands on its `PATH`, and whether `expanse-client`'s module (`sdsc`) is loaded in it, is unconfirmed.
- **`expanse-client`.**
  The observer runs `expanse-client user -r expanse` and `-r expanse_gpu`, the invocation and the resource names the public user guide shows, and recognises only the table the guide shows (rows drawn with the vertical bar, a header naming PROJECT, USED, AVAILABLE and USED BY PROJECT, a whole number in each of the last three columns).
  Anything else is reported as the lines that were printed.
  Whether the client works non-interactively under a Tapis job (the guide lists an authentication flag) and what it prints there is unconfirmed, and so is the unit of the figures.
- **The Slurm formats.**
  The observer's fixtures follow the formats the Slurm manual documents for `sinfo -o`, `squeue -o` and `sacct --format`; none was captured from a cluster.
  `squeue --start` is asked with an explicit format (`%i|%P|%T|%S`), which the manual says overrides the one `--start` implies; which jobs `sacct -X -S -E` selects for a day, and the time zone of its times, are also unconfirmed.
  A state word or reason the observer has not seen is carried as printed, and a row in another format is counted, never guessed at.
- **The receiving route and the observation contract.**
  `POST` of a `nemar.observation.v1` document with a bearer token is NEMAR's, built in the gateway epic; the observer is tested against a stand-in of it, and its header comment is the only description of the document today.
  Phase A7 must pin the contract in BOTH repositories: a JSON Schema in nemar-cli's shared contract, a run identifier that is the Tapis job's UUID so a retried post is idempotent, a per-run token separate from every job's token, a 2xx answer with no redirect, and `raw_lines` treated as untrusted text.
  This pull request is safe to merge before that, because the observer does nothing until NEMAR submits it.

### Also listed above

Whether `--no-mount bind-paths` has its effect, that the PID namespace exists, and whether the site allows `--net --network none` are under [The node](#the-node); the host Python, outbound HTTPS from the nodes, the local dataset root layout and the SIF's compatibility with the cluster's Apptainer are there and under [The runtime image](#the-runtime-image-nemar-python).
