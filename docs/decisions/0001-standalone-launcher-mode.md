# 0001: The launcher's standalone mode

Status: accepted (epic nemarOrg/nemar-cli#2324, Phase 6, nemarOrg/nemar-process#67; ADR 0134 decision 8 in nemar-cli).

## Context

A person should be able to run a NEMAR recipe, or their own job written to the same contract, on their own Linux machine and get the same products, checkpoint counts and provenance, with no NEMAR service involved.
The launcher already isolates a job's container and reads back what it left; the service path also holds a job token, fetches code and data from NEMAR, posts progress and uploads results.

## Decision

1. **A separate entry, not a flag.**
   `launcher/nemar_launch/local.py` (`main(argv, environ)`) behind the POSIX wrapper `nemar-launch-local` at the zip's root.
   The service entry takes no command line and Tapis and a runner can set environment variables, so neither a flag nor a variable could select the mode safely.
   The entry builds its environment from an allowlist and never reads `NEMAR_LAUNCH_SPEC`, a lease or a scheduler variable.
2. **One container phase.**
   `__main__._do_work` takes its inputs through `Steps` (code, data, image, progress posting, the reporter's cadence, the engine's pull policy); the service and the standalone entry give their own.
   A `spec.Job` is what the container phase needs; only a `spec.Spec` carries an API address, a token and limits, and the functions that use them take a `Spec`, so the type checker proves a local job cannot reach them.
   The isolation is identical; the documented differences are Docker's `--pull=never`, a random runner label, and SIGINT and SIGHUP stopping the job.
3. **Schemas in nemar-cli.**
   The local launch specification, the recipe progress file and the results report are written in nemar-cli (`shared/contract/`, TypeScript and JSON Schema) and copied here byte for byte under the progress pin ([contracts.md](../contracts.md)).
   Each of the launch specification and the local one has a key the other refuses.
4. **A fence and one window.**
   An audit hook (`fence.py`) refuses sockets, lookups, requests and every program but the container runtime's own verbs for the whole run.
   Only `--pull` opens one window, around the image's verification and pull, allowing exactly the cosign, `docker pull` and `apptainer pull` command lines computed then and, for the SIF manifest, the registry's host and two addresses.
   `--pull` accepts only NEMAR's two signed repositories, verifies with the runner's own policy (`verify.py`, held equal to the runner's by a test), keeps cosign's trust root in a per-run home inside the scratch, and caches nothing.
   Without `--pull` nothing is pulled.
   The network of a child process and of the Docker daemon is outside the fence, a documented limit.
5. **The report.**
   `<out>/launcher/results-report.json` (`report.py`): what ran, on what and what came out, with the checkpoint counts by the backend's reconcile rules over a recordings table whose digest is the manifest's.
   A problem is a fixed word inside it and never changes the exit; runners and Tapis write no report in this phase.

## Consequences

The zip's digest changes (two files at its root).
No failure code or detail word was added: a refused verification is `image_unverified`, an image not pulled is `image_missing`.
A docs.nemar.org guide for this mode follows once the epic merges.
