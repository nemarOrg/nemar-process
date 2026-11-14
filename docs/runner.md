# The runner: `nemar-runner`

`nemar-runner` is the program a workstation runs to take jobs from NEMAR.
It exchanges its secret for a short-lived token, claims jobs, runs each one through the launcher ([launcher.md](launcher.md)) in a directory of its own under Apptainer, Docker or Podman, renews the lease while the job runs, enforces the deadline and memory, completes the job with the usage it measured, and always deletes the job's directory.
It opens no port: it polls NEMAR over HTTPS.
It is standard library only and runs on Python 3.9 or newer, as the launcher does.

Abbreviations used here: parent-death signal (PDEATHSIG, the Linux `prctl` option `PR_SET_PDEATHSIG`), application programming interface (API), JavaScript Object Notation (JSON), Singularity Image Format (SIF), OCI Registry As Storage (ORAS), GitHub Container Registry (GHCR), process group (PG), resident set size (RSS), control group (cgroup), continuous integration (CI).

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
SIGTERM, SIGHUP or SIGINT stops claiming, terminates the jobs still running (SIGTERM to each job's process group, SIGKILL 30 seconds later), completes them, deletes their directories, and exits.

| Exit | Meaning |
|---|---|
| 0 | done |
| 1 | a failure (`doctor` found a problem, or `once` could not claim) |
| 2 | a configuration or secret the runner will not start with |
| 3 | NEMAR will not serve this runner: its secret is not accepted (the jobs it holds are stopped), or it is older than the Worker's `min_runner_version` or speaks another protocol (`runner_outdated`, `protocol_unsupported`: it stops claiming, lets the jobs it holds finish, then exits) |

## Configuration

A JSON file at `$XDG_CONFIG_HOME/nemar-runner/config.json` (default `~/.config/nemar-runner/config.json`; an empty or relative `XDG_CONFIG_HOME` is ignored, as the specification says).
Every key but `launcher_dir`, `default_job_memory_mb` and `sif_pull` is required, and an unknown key is refused.

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
| `sif_pull` | optional boolean, default `false`: when false the runner never pulls an image and a missing image fails the job as `image_missing` (the operator places the image and verifies it); true is for development only ([The image](#the-image)) |
| `launcher_dir` | optional: the directory that holds the `nemar_launch` package and its `schema` directory (default: the `launcher` directory of this checkout) |

The runner refuses to start when the configuration file or the secret file can be reached by group or others (any group or other permission bit: use mode 0600), is not the user's own regular file (a link is refused), or when `root` is not a directory the user owns, or can be written by group or others.
A refusal names the key and never a value.

**Run the runner as a dedicated account.**
The runner's secret is in its process memory and its job directories hold the job tokens' launch values, and processes of the same account can read the secret through `/proc/<pid>/mem` where `kernel.yama.ptrace_scope` is 0.
Nothing else that runs jobs, or that you do not trust, may run as that account.

**What it will not run.**
At start the runner also refuses a `launcher_dir` with any file or directory that is a link, is owned by someone other than the user or root, or that group or others can write; a `sif_root` that is not the user's own directory or that group or others can write; and a `python3` on the launch PATH (the first of `/usr/local/bin:/usr/bin:/bin`) that, or whose directory, group or others can write, that is owned by someone else, or that is older than Python 3.9.
Engine helpers (`ps`, `rm`, `--version`, the preflight) run with `PATH` and `LANG` only, so the engine's default socket is the one used; a rootless engine, a non-default `DOCKER_HOST` or `XDG_RUNTIME_DIR` is unsupported (the launcher does not support a rootless daemon either).

## What a job gets

Under `<root>/jobs/<job id>-<lease id>/` (mode 0700, named by the lease too, so a finished run or a second claim of the same job can never delete another's directory): `out/` (the launcher's output directory, `NEMAR_OUTPUT_DIR`), `scratch/` and `home/`.
Nothing is shared between jobs.
The directory is deleted in a `finally`, whatever happened, and only then is the job completed, so the completion's `scratch_discarded` is what happened.
The log line is `scratch_discarded` with the job id and the bytes the directory held.
The deletion walks the tree by directory descriptor and never follows a link, so a directory swapped for a link while it runs cannot steer it out of the tree.
When the runner starts it stops what a dead run left (see [When the runner dies](#when-the-runner-dies)) and then deletes every leftover under `<root>/jobs/` and `<root>/pids/` and logs each job directory (`leftover_deleted`).
A claim of a job id that this runner is still running is skipped (`duplicate_claim_skipped`).
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
| `NEMAR_RUNNER_LABEL` | Docker and Podman only: the runner's registration id, `rnr_` and 24 lowercase hexadecimal digits, which the launcher puts on every container as `nemar.runner` | `GET /compute/runners/me`, read at start |
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
With `sif_pull` false (the default) the runner never pulls: a missing image ends the job without launching, exit 13, `runtime_error`, detail `image_missing`, zero usage, and the operator places and verifies the image.

With `sif_pull` true (development only) the runner runs `apptainer pull <sif_root>/.<hex>.sif.<pid>.part oras://ghcr.io/nemarorg/nemar-python-sif:sha256-<hex>` (`singularity` when there is no `apptainer`), with its cache and temporary directories inside the job's `scratch/`, then renames the part file into place.
The pull is by a mutable tag and the runner checks no digest and no signature of what it pulled: `docs/images.md` signs only the digest, so a pull by tag is trusted as far as the registry is.
The part name is unique to the process, a lock file beside the image (`.<hex>.sif.lock`, an advisory lock held across jobs and runner processes) makes two jobs or two runners pull one image once, a part file of a dead run is deleted under the lock, and waiting for the lock stops when the job is stopped or its deadline passes.
A failed pull, or one that writes nothing, ends the job as `image_missing`.
A pull happens before any heartbeat: if the image took longer than the lease less 60 seconds, the runner abandons that job without completing it (`claim_abandoned_during_pull`), so the lease expires and the job waits again, unstarted; the pulled file stays.
Docker and Podman pull by digest inside the launcher, after `started`, so the runner does nothing for them.

The readiness probe of [`doctor`](#doctor) runs `apptainer exec --containall --cleanenv --no-home --no-mount bind-paths --memory 64M --cpus 1 <image> true`, so the image never runs with the host's home directory or environment (and the runner's secret).

## While a job runs

- **Heartbeat.**
  A thread starts only after the launcher process is spawned (never before or during an image pull: the contract has no other phase), sends `started` at once and then renews the lease every 60 seconds; `uploading` is sent at once when the launcher has written `<job>/out/launcher/result.json` (the path the launcher uses, `launcher/` under its output directory).
  It goes on until the completion is sent or given up: removing containers, deleting the directory and retrying the completion can outlast the 600-second lease.
  A heartbeat that fails is retried and the job goes on, and the thread survives any error.
- **Deadline.**
  SIGTERM to the process group at `NEMAR_JOB_END_TIME` plus 60 seconds, SIGKILL 30 seconds after any SIGTERM.
  The wall clock is read once, when the job starts, and the deadline is kept on the monotonic clock: a step of the system clock cannot move it.
  A signal is sent before the line that says so is logged, and the log never raises (a full disk or a closed pipe is counted in a dropped-lines counter), so a failing log cannot leave a job running.
- **Memory.**
  Without enforced limits, a watchdog sums the resident memory of the whole process tree (from `ps`) every 10 seconds and sends SIGTERM above the job's `memory_mb`.
  It counts resident memory only: shared memory and a `tmpfs` escape it, and under Apptainer a job claimed without `memory_mb` has no memory bound at all.
  With `NEMAR_ENFORCE_LIMITS=1` the container runtime enforces it and the watchdog stands down.
- **Cancel.**
  A heartbeat answer with `cancel` set: SIGTERM, 30 seconds, SIGKILL, then the job is completed.
- **Lost lease.**
  A heartbeat answer of `lease_lost` or `job_not_claimed` (HTTP 409 and 404), or of `runner_unknown` (the runner was revoked): the same stop, then the directory is deleted and nothing is completed.
- **After the process exits** the whole process group is sent SIGKILL, and whatever goes wrong while supervising (any exception) also ends in SIGKILL to the group.

### When the runner dies

What is guaranteed, and what is not:

- On Linux the launcher child is started with `prctl(PR_SET_PDEATHSIG, SIGTERM)` (through `ctypes`), so the kernel sends it SIGTERM when the runner's thread that started it dies, SIGKILL of the runner included; the launcher forwards SIGTERM to its container.
  Elsewhere the runner logs `pdeathsig_unavailable` for each job and has no such guarantee.
- Each launcher's pid, process-group id and start time (field 22 of `/proc/<pid>/stat`, the `ps` start time elsewhere) go into `<root>/pids/<job id>.json` before the heartbeat starts and are deleted when the job ends.
  When the runner starts it sends SIGTERM, and 30 seconds later SIGKILL, to every recorded group whose leader still has the recorded start time (a reused pid is left alone); then it removes this runner's labeled containers; only then it deletes the leftover job directories and pid files.
- SIGHUP is handled exactly as SIGTERM.
- After a normal end, a cancel, a lost lease or a deadline kill, the process group is sent SIGKILL.

What is not guaranteed: a job that leaves the launcher's process group (a process that calls `setsid`) is not reached by a group signal, and the launcher has no deadline timer of its own, so a launcher that is alive but whose runner is gone keeps running until the next runner start or the kernel's death signal (outline step L1 gives the launcher a timer).

### Containers (Docker and Podman)

The launcher labels each container `nemar.job=<job id>` and `nemar.runner=<runner id>` (the id the runner passes as `NEMAR_RUNNER_LABEL`).
When a job ends, however it ends (a lost lease, a cancel, a deadline kill, any exit of the launcher, or an error in the runner), the runner runs `<engine> ps -aq --filter label=nemar.runner=<runner id> --filter label=nemar.job=<job id>` and then `<engine> rm --force --volumes <ids>` when any id is listed.
When the runner starts it does the same with `--filter label=nemar.runner=<runner id> --filter label=nemar.job`, which removes the containers a dead run of this runner left and never another runner's, on a daemon the runners share.
Each removal is logged as `container_removed` with the job id (none, for the start-up sweep) and the short container id.
Without the registration id nothing is removed.
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
An answer whose `min_runner_version` is newer than the runner still has its jobs run (they are leased to this runner, and dropping them would spend a claim each); the runner then drains and exits 3.
A 426 (`runner_outdated`, `protocol_unsupported`) at claim time drains the same way: no more claims, the jobs in hand finish; only a refused secret (`runner_unknown`) stops the jobs.
An exception of any kind around a claim is logged as a fixed word (`claim_error`) and the loop backs off; the runner never exits with a job thread alive.
A failed poll (a timeout, a 5xx, a dropped connection) is logged and the loop waits longer, up to five minutes, before it asks again.
A `complete` that fails the same way is retried with a growing pause; if it never gets through, NEMAR's lease sweep decides the job.

The token is exchanged at the start, and again 60 seconds before it expires or when NEMAR answers `runner_token_expired`, once however many jobs ask at the same time; it is kept in memory only.
A timestamp that has the right shape and a date the calendar refuses (`2026-02-30T00:00:00Z`) is a protocol error, not a crash.
Under Docker or Podman the runner reads its registration id from `GET /compute/runners/me` at start (the answer's `runner_id`) and refuses to start if it cannot.
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
| `launcher python` | a `python3` is on that path, neither it nor its directory is writable by group or others, and it is Python 3.9 or newer |
| `launcher` | `launcher_dir` holds the `nemar_launch` package |
| `launcher_dir trusted` | nothing under `launcher_dir` is a link, owned by someone else, or writable by group or others |
| `sif_root writable` | `sif_root` is the user's, not writable by others, and a file can be made in it |
| `egress to api_base` | the host answers HTTP |
| `token exchange` | the secret buys a token |
| `limits` (Apptainer only) | `apptainer exec --memory 64M --cpus 1 <an image in sif_root> true` works, which proves cgroup delegation |
| `engine limits enforceable`, `engine mode accepted` (Docker and Podman only) | the two answers of the launcher's own engine preflight (`nemar_launch.runtime.engine_preflight(engine_path, engine_name)`, called in a child process run as the launcher is), so `doctor` and the launcher cannot disagree about an engine; rootless Docker without enforceable limits answers no |

Exit 0 only if every line but the Apptainer `limits` line is yes: a host that cannot enforce limits still runs jobs, with the watchdog.
`run` and `once` run the same probe the first time an image is available and set `NEMAR_ENFORCE_LIMITS` from it; Docker and Podman are never claimed, because the launcher owns their flags and checks them with the preflight.

## Developing

```bash
uv run --no-project --python 3.9 --with pytest --with pytest-timeout pytest tests/runner -q     # the runner on its floor (each test is limited to two minutes)
uv run pytest tests/contract/test_runner_drift.py -q                      # against the pinned nemar-cli commit
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run ty check --python-version 3.9 launcher observer runner
```

The tests run real child processes and real HTTP; NEMAR's Worker is a recording stand-in that answers fixtures and checks each request body against the pinned schema, and the container runtime is a stand-in executable where a test is about how it is called.
The drift test needs a nemar-cli checkout that has the pinned commit (`NEMAR_CLI_RUNNER_DIR`, or a `nemar-cli` directory beside this checkout), and fails instead of skipping when `NEMAR_REQUIRE_RUNNER_DRIFT=1`.
