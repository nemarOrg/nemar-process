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
- **PID namespace verification.** The launcher fails closed: with no way to verify the namespace it refuses before any user code (`pid_namespace_unavailable`). It reads `allow pid ns` from the configuration it can find (the `SYSCONFDIR` that `<runtime> buildcfg` prints, beside the install prefix, `/usr/local/etc`, `/etc`) and, for `runtime: python`, runs a preflight probe in the job's image, which proves the namespace even where no configuration is found.
  For `runtime: image` (a user's image, no Python to count on) only the configuration can verify it, so a user image on a node whose configuration is somewhere else, or cannot be read, is refused and not run.
  Whether `buildcfg` prints `SYSCONFDIR` on the node's runtime is unconfirmed (CI checks only that, where it answers, the directory exists); if it does not, the usual places are still tried.
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

- No image is built or pushed (the runtime image is phase 3); the container tests use a small public image.
- No Tapis application or system definition is written, validated or registered, and no observer exists (phase 6).
- A symlink or special file that cannot be unlinked even after the directory's permissions are widened, a directory that cannot be listed even after the same, and an entry that vanishes between the directory listing and its `lstat`, have no test: an unprivileged test cannot make any of them on demand (the walk records each as `outputs_unreadable` or skips it, and the code paths are read, not run).
