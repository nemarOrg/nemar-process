# The runner: `nemar-runner`

`nemar-runner` is the program a workstation runs to take jobs from NEMAR.
It exchanges its secret for a short-lived token, claims jobs, runs each one through the launcher ([launcher.md](launcher.md)) in a directory of its own under Apptainer, Docker or Podman, renews the lease while the job runs, enforces the deadline and memory, completes the job with the usage it measured, and always deletes the job's directory.
It opens no port: it polls NEMAR over HTTPS.
It is standard library only and runs on Python 3.9 or newer, as the launcher does.

Abbreviations used here: application programming interface (API), JavaScript Object Notation (JSON), Singularity Image Format (SIF), OCI Registry As Storage (ORAS), GitHub Container Registry (GHCR), process group (PG), resident set size (RSS), control group (cgroup), continuous integration (CI).

The protocol is `nemar.runner.v1`.
Its words and shapes are written in nemar-cli (`shared/contract/compute-runner.ts` and `compute-runner.schema.json`) and copied here: `runner/schema/compute-runner.v1.json` is a byte copy, pinned in `tests/contract/nemar-cli-runner-pin.json` and held by `tests/contract/test_runner_drift.py` ([contracts.md](contracts.md)).
This page is about the program; the integration with NEMAR's real routes is proven in the card that owns it, against the local stack, not here.

## Running it

```bash
cd runner && python3 -m nemar_runner doctor      # check this host, one yes or no a line
cd runner && python3 -m nemar_runner once        # claim once, run what was claimed, exit
cd runner && python3 -m nemar_runner run         # claim until stopped (SIGTERM or SIGINT)
```

`--config <file>` names another configuration file.
SIGTERM or SIGINT stops claiming, terminates the jobs still running (SIGTERM to each job's process group, SIGKILL 30 seconds later), completes them, deletes their directories, and exits.

| Exit | Meaning |
|---|---|
| 0 | done |
| 1 | a failure (`doctor` found a problem, or `once` could not claim) |
| 2 | a configuration or secret the runner will not start with |
| 3 | NEMAR will not serve this runner: its secret is not accepted, it is older than the Worker's `min_runner_version`, or it speaks another protocol |

## Configuration

A JSON file at `$XDG_CONFIG_HOME/nemar-runner/config.json` (default `~/.config/nemar-runner/config.json`; an empty or relative `XDG_CONFIG_HOME` is ignored, as the specification says).
Every key but `launcher_dir` and `default_job_memory_mb` is required, and an unknown key is refused.

| Key | Meaning |
|---|---|
| `api_base` | NEMAR's API, `https://...` (`http://` only on loopback) |
| `secret_file` | the file that holds the runner secret, `nmr_` and 64 lowercase hexadecimal digits |
| `root` | the runner's working directory: jobs, the log and the lock |
| `sif_root` | where Apptainer images live, `<sif_root>/<hex>.sif` |
| `runtime` | `apptainer`, `docker` or `podman` |
| `slots` | how many jobs run at once, 1 to 32 |
| `arch` | `amd64` or `arm64` |
| `max_scratch_gb` | the runner does not claim while free space under `root` is below this |
| `default_job_memory_mb` | a positive whole number of megabytes: the memory size of a job that the claim gives none, REQUIRED when `runtime` is `docker` or `podman` (the runner refuses to start without it), optional under Apptainer, where it is never applied |
| `launcher_dir` | optional: the directory that holds the `nemar_launch` package and its `schema` directory (default: the `launcher` directory of this checkout) |

The runner refuses to start when the configuration file or the secret file can be reached by group or others (any group or other permission bit: use mode 0600), is not the user's own regular file (a link is refused), or when `root` is not a directory the user owns, or can be written by group or others.
A refusal names the key and never a value.

## What a job gets

Under `<root>/jobs/<job id>/` (mode 0700): `out/` (the launcher's output directory, `NEMAR_OUTPUT_DIR`), `scratch/` and `home/`.
Nothing is shared between jobs.
The directory is deleted in a `finally`, whatever happened, and only then is the job completed, so the completion's `scratch_discarded` is what happened.
The log line is `scratch_discarded` with the job id and the bytes the directory held.
When the runner starts it deletes every leftover under `<root>/jobs/` and logs each one (`leftover_deleted`).
One runner per `root`: a second one is refused while the first holds `<root>/runner.lock`.

### The launcher's environment

The launcher is started as `python3 -I -S -c '<boot line>' <launcher directory>` (the boot line of [launcher.md](launcher.md#outside-tapis-and-slurm)) in a new process group, with this environment and nothing else:

| Key | Value | Source |
|---|---|---|
| `NEMAR_LAUNCH_SPEC` | the claim's `launch_spec_env` | the claim answer |
| `NEMAR_SIF_ROOT` | `sif_root` | the configuration |
| `NEMAR_OUTPUT_DIR` | `<job>/out` | the runner |
| `NEMAR_SCRATCH_DIR` | `<job>/scratch` | the runner |
| `NEMAR_JOB_END_TIME` | the time the claim answer arrived plus the job's `minutes`, in whole seconds since the epoch | the claim answer and the clock |
| `NEMAR_LEASE_ID` | the job's `lease_id` | the claim answer |
| `NEMAR_CONTAINER_RUNTIME` | `runtime` | the configuration |
| `NEMAR_ENFORCE_LIMITS` | `1` only after `doctor`'s probe has passed in this process (Apptainer only), else `0`; under Docker and Podman the launcher always applies limits and does not read it | the probe |
| `NEMAR_CPUS` | the job's `cores`, always | the claim answer |
| `NEMAR_MEMORY_MB` | the job's `memory_mb`; under Docker or Podman always, `default_job_memory_mb` when the claim's is null (the log line is `memory_default_applied job=<id> mb=<n>`); under Apptainer only when the claim's is not null | the claim answer, the configuration |
| `HOME` | `<job>/home` | the runner |
| `PATH` | `/usr/local/bin:/usr/bin:/bin` | fixed |
| `LANG` | `C.UTF-8` | fixed |

The launch specification carries no cores or memory, by design, so the runner passes both from the claim answer; the launcher fails closed when `NEMAR_ENFORCE_LIMITS=1` has no usable `NEMAR_CPUS`.
The launcher fails closed under an engine without `NEMAR_MEMORY_MB`, because the runner's watchdog sees only the engine's client process there.
The memory watchdog below uses the same value.
Nothing of the runner's own environment reaches the launcher.

### The launch value

A claim's `launch_spec_env` is checked by the schema for its alphabet and length only, so the runner decodes it with the seven rules of the launcher's decoder (`runner/nemar_runner/wire.py`, replayed against the launcher's table of wire cases) before it does anything for the job.
A value that fails ends the job without launching: exit 13, `runtime_error`, detail `invalid_spec`, the answer the launcher would give.
So does a specification whose `job_id` is not the claimed job's, or whose image file is not the claim's.
The decoded text holds the job token and is never logged.

### The image

For Apptainer the file `<sif_root>/<hex>.sif` must exist, where `<hex>` is the image digest.
If it does not, the runner runs `apptainer pull <sif_root>/.<hex>.sif.part oras://ghcr.io/nemarorg/nemar-python-sif:sha256-<hex>` (`singularity` is used when there is no `apptainer`) with its cache and temporary directories inside the job's `scratch/`, then renames the part file into place, so a half-written image is never run.
Two jobs for one image pull it once.
A failed pull, or one that writes nothing, ends the job without launching: exit 13, `runtime_error`, detail `image_missing`, zero usage.
Docker and Podman pull by digest inside the launcher, so the runner does nothing for them.

## While a job runs

- **Heartbeat.**
  A thread sends `started` and then renews the lease every 60 seconds; `uploading` is sent at once when the launcher has written `<job>/out/launcher/result.json` (the path the launcher uses, `launcher/` under its output directory).
  The thread starts before the image is pulled, not after the launcher is spawned, because a pull of a large image can outlast the 600-second lease and no other signal renews it.
  A heartbeat that fails is retried and the job goes on.
- **Deadline.**
  SIGTERM to the process group at `NEMAR_JOB_END_TIME` plus 60 seconds, SIGKILL 30 seconds after any SIGTERM.
- **Memory.**
  Without enforced limits, a watchdog sums the resident memory of the whole process tree (from `ps`) every 10 seconds and sends SIGTERM above the job's `memory_mb`.
  With `NEMAR_ENFORCE_LIMITS=1` the container runtime enforces it and the watchdog stands down.
- **Cancel.**
  A heartbeat answer with `cancel` set: SIGTERM, 30 seconds, SIGKILL, then the job is completed.
- **Lost lease.**
  A heartbeat answer of `lease_lost` or `job_not_claimed` (HTTP 409 and 404), or of `runner_unknown` (the runner was revoked): the same stop, then the directory is deleted and nothing is completed.
- **After the process exits** the whole process group is sent SIGKILL, so nothing of the job outlives it.

### Containers (Docker and Podman)

The launcher labels each container `nemar.job=<job id>`.
When a job ends, however it ends (a lost lease, a cancel, a deadline kill, any exit of the launcher, or an error in the runner), the runner runs `<engine> ps -aq --filter label=nemar.job=<job id>` and then `<engine> rm --force --volumes <ids>` when any id is listed.
When the runner starts it does the same with `--filter label=nemar.job`, which removes every container a dead run of this account left.
Each removal is logged as `container_removed` with the job id (none, for the start-up sweep) and the short container id.
Ids the engine prints are checked to be hexadecimal before they reach `rm`; a failing engine is logged (`container_cleanup_failed`) and never stops the job's completion.
Under Apptainer nothing is done.

### Usage

`os.wait4` gives the wall time, the CPU time (user plus system) and the largest resident set of the child.
Seconds are rounded to the nearest whole second and memory up to a whole megabyte.
Under Docker or Podman the container's processes are the runtime daemon's, not the launcher's children, so `usage_partial` is true and the log has `usage_partial runtime=<runtime>`.
The completion's `exit_number` is the launcher's exit status (128 plus the signal for a signal death); `launcher_failure_code` and `launcher_detail` are read from `result.json` only when it holds one of the seven codes and a fixed-word detail.

## The claim

`run` polls every 15 seconds while a slot is free, and `once` claims once and waits.
A claim asks for as many jobs as there are free slots, and the answer is refused whole (nothing runs) when it holds more jobs than that, a job twice, or anything the schema forbids: the schema cannot say `jobs.length <= free_slots`, so the runner does.
An answer whose `min_runner_version` is newer than the runner stops it (exit 3) without running the jobs.
A failed poll (a timeout, a 5xx, a dropped connection) is logged and the loop waits longer, up to five minutes, before it asks again.
A `complete` that fails the same way is retried with a growing pause; if it never gets through, NEMAR's lease sweep decides the job.

The token is exchanged at the start, and again 60 seconds before it expires or when NEMAR answers `runner_token_expired`; it is kept in memory only.
Redirects are never followed, so a credential cannot be sent elsewhere.

## Logs

JSON lines, one object per line, to `<root>/runner.log` (mode 0600) and to standard error.
A line holds job ids, counts, fixed words and exception class names: never the secret, a token, a lease, a presigned address, a launch value, or anything the job or the launcher wrote.
Every string field also passes a redaction of those shapes, as a second line of defense.

## `doctor`

One line each, `<check>: yes` or `no`, with a fixed phrase after a no:

| Check | Yes means |
|---|---|
| `config permissions`, `config` | the configuration file is private and valid |
| `secret permissions`, `secret` | the secret file is private and holds a runner secret |
| `root owned by the user` | `root` is the user's and not writable by others |
| `runtime <name>` | the runtime's executable is on `/usr/local/bin:/usr/bin:/bin` and answers `--version` (the version is shown) |
| `launcher python` | a `python3` is on that path |
| `launcher` | `launcher_dir` holds the `nemar_launch` package |
| `sif_root writable` | `sif_root` is the user's and a file can be made in it |
| `egress to api_base` | the host answers HTTP |
| `token exchange` | the secret buys a token |
| `limits` (Apptainer only) | `apptainer exec --memory 64M --cpus 1 <an image in sif_root> true` works, which proves cgroup delegation |
| `engine limits enforceable`, `engine mode accepted` (Docker and Podman only) | the two answers of the launcher's own engine preflight (`nemar_launch.runtime.engine_preflight(engine_path, engine_name)`, called in a child process run as the launcher is), so `doctor` and the launcher cannot disagree about an engine; rootless Docker without enforceable limits answers no |

Exit 0 only if every line but the Apptainer `limits` line is yes: a host that cannot enforce limits still runs jobs, with the watchdog.
`run` and `once` run the same probe the first time an image is available and set `NEMAR_ENFORCE_LIMITS` from it; Docker and Podman are never claimed, because the launcher owns their flags and checks them with the preflight.

## Developing

```bash
uv run --no-project --python 3.9 --with pytest pytest tests/runner -q     # the runner on its floor
uv run pytest tests/contract/test_runner_drift.py -q                      # against the pinned nemar-cli commit
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run ty check --python-version 3.9 launcher observer runner
```

The tests run real child processes and real HTTP; NEMAR's Worker is a recording stand-in that answers fixtures and checks each request body against the pinned schema, and the container runtime is a stand-in executable where a test is about how it is called.
The drift test needs a nemar-cli checkout that has the pinned commit (`NEMAR_CLI_RUNNER_DIR`, or a `nemar-cli` directory beside this checkout), and fails instead of skipping when `NEMAR_REQUIRE_RUNNER_DRIFT=1`.
