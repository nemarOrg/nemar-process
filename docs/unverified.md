# What is not verified yet

The launcher is tested against local stand-ins and, in CI, against a real Apptainer on a Linux runner.
It has not run on a compute node.
These are the facts it depends on that nobody has confirmed on the node or against NEMAR's Worker, in the order they would bite.

## The node

- **Host Python.** The launcher needs Python 3.9 or newer on the node, outside the container.
  If the node has only an older `python3`, the wrapper exits 13 with a message and the application definition has to name an interpreter (`NEMAR_HOST_PYTHON`).
  This is an escalation to the owner, not something to work around in the launcher.
- **Outbound HTTPS from compute nodes** to NEMAR's API and to the object storage endpoint the presigned addresses point at.
  It was measured as available in October 2026; it is still a question for the center that runs the nodes.
- **The container runtime's behaviour on the node.** CI uses the community Apptainer; the node may run a commercial Singularity.
  `--containall`, `--cleanenv`, `--no-home`, `--env`, `--bind ...:ro` and `--nv` are common to both, but the isolation tests have not run there.
  The runtime's site configuration may bind paths into every container; the launcher does not disable them (`--no-mount`), so check the site's `apptainer.conf` or `singularity.conf` for `bind path` entries.
- **How `--env` is parsed.** Apptainer 1.5.5 reads the flag as a CSV record, and the launcher quotes values accordingly; whether the commercial Singularity on the node parses it the same way is unconfirmed (a value with a comma is the case to try).
- **GPU visibility.** `CUDA_VISIBLE_DEVICES` is not passed into the container (the environment is the allowlist only), so a GPU job relies on the scheduler's device cgroup to show the container only the GPUs it was given.
- **The image file.** `image.sif` is resolved relative to `NEMAR_SIF_ROOT`, else the working directory.
  That variable is this phase's choice, not nemar-cli's (it is not in `LAUNCHER_ENV`): the application definition has to set it, or stage the file next to the job.
- **Slurm's environment.** The launcher reads `SLURM_JOB_END_TIME` (the time limit rule), `SLURM_JOB_ID` and `SLURM_TMPDIR`.
  Whether a Tapis ZIP-runtime job sees them, and how long Slurm waits between SIGTERM and SIGKILL (the upload must finish inside it), is unconfirmed.
- **Tapis.** The ZIP runtime runs `./tapisjob_app.sh` and reads `tapisjob.exitcode`; the launcher writes that file itself and exits with the same number.
  Whether the runtime keeps the script's mode bits, and what it does with a non-zero exit, is unconfirmed.

## NEMAR's data and storage

- **The local dataset root layout.** The launcher binds `<NEMAR_LOCAL_DATASETS_ROOT>/<id>/<version>/` when every selected file there matches the manifest.
  Whether NEMAR's copy on the node is laid out that way, and holds plain files and not symlinks into an annex object store, is unconfirmed; a symlink never matches, so a git-annex checkout would always fall back to download.
- **Presigned multipart upload to the results bucket.** A single PUT per file is the only upload the launcher does, so a single result file is limited to 5 GB until multipart upload through presigned addresses is verified on R2.
- **What the presigned addresses require.** The launcher PUTs with `Content-Type: application/octet-stream`, a `Content-Length`, and no token; if the Worker signs other headers, the PUT fails with a 403 and is re-requested once.
  It sends a `nemar-launch/<version>` `User-Agent` on every request, in case a content delivery network rejects the default.
- **The Worker's responses.** The outputs and bundle routes do not exist yet; the stand-ins follow the schema's `$defs`.
  The error body is read for a `code` word only (`results_too_large`, `results_quota_exceeded`, `too_many_outputs`, `bad_state`, ...), and the launcher posts `{}` as the body of the bundle and complete routes.
- **Details that are NEMAR-caused but exit 13.** `invalid_spec`, `scratch_in_output`, `scratch_unavailable`, `no_container_runtime`, `image_missing`, `runtime_unusable` and `internal_error` are `runtime_error` because the launcher has only seven codes.
  NEMAR's side decides whether a job failing with one of those details is refunded.

## Not done in this phase

- No image is built or pushed (the runtime image is phase 3); the container tests use a small public image.
- No Tapis application or system definition is written, validated or registered, and no observer exists (phase 6).
