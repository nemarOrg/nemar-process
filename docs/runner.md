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
Every directory of the launch PATH up to the one holding `python3` (and the runtime) must be the user's or root's and not writable by group or others, the `python3` path is resolved once at start and pinned for every job, and the runtime binary gets the same check.
Engine helpers (`ps`, `inspect`, `rm`, `--version`, the preflight) run with `PATH` and `LANG` only, so the engine's default socket is the one used; a rootless engine, a non-default `DOCKER_HOST` or `XDG_RUNTIME_DIR` is unsupported (the launcher does not support a rootless daemon either).

## What a job gets

Under `<root>/jobs/<job id>-<lease id>/` (mode 0700, named by the lease too, so a finished run or a second claim of the same job can never delete another's directory): `out/` (the launcher's output directory, `NEMAR_OUTPUT_DIR`), `scratch/` and `home/`.
Nothing is shared between jobs.
The directory is deleted in a `finally`, whatever happened, and only then is the job completed, so the completion's `scratch_discarded` is what happened.
The log line is `scratch_discarded` with the job id and the bytes the directory held.
The deletion walks the tree by directory descriptor and never follows a link, so a directory swapped for a link while it runs cannot steer it out of the tree.
The launcher's recorded pid, group and start time are believed only when the record file and the `pids` directory are the user's own and not writable by others (a link is never followed), the group is the pid's own, `getpgid` agrees, the start time matches and the boot id is the same.
When the runner starts, first of all (before any trust check or call to NEMAR can refuse the start), it stops what a dead run left (see [When the runner dies](#when-the-runner-dies)) and then deletes every leftover under `<root>/jobs/` and `<root>/pids/` and logs each job directory (`leftover_deleted`).
A claim of a job id that this runner still has a run of (a second lease of it) is not dropped, because each claim spends one of the job's three: the new run waits, bounded, for the old one to end (`duplicate_claim_waiting`) and then goes ahead, or is given up (`duplicate_claim_abandoned`).
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
  The launcher stops its own container at `NEMAR_JOB_END_TIME` itself (SIGTERM, and SIGKILL 10 seconds later), so the runner's signal 60 seconds after normally finds the container gone and is the backstop for a launcher that is stuck outside its container.
  The wall clock is read once, when the job starts, and the runner's deadline is kept on the monotonic clock: a step of the system clock cannot move it.
  The launcher's own timer acts first and reads the wall clock, for the end time and for its 10 seconds of grace: a step forward ends the job early, with `time_limit`, and a step backward delays the launcher's stop, which the runner's monotonic deadline plus its margin (or, under Tapis, the scheduler) still bounds.
  A signal is sent before the line that says so is logged, and the log never raises (a full disk or a closed pipe is counted in a dropped-lines counter), so a failing log cannot leave a job running.
- **Memory.**
  Without enforced limits, a watchdog sums the resident memory of the whole process tree (from `ps`) every 10 seconds and sends SIGTERM above the job's `memory_mb`.
  It counts resident memory only: shared memory and a `tmpfs` escape it, and under Apptainer a job claimed without `memory_mb` has no memory bound at all.
  With `NEMAR_ENFORCE_LIMITS=1` the container runtime enforces it and the watchdog stands down.
- **Cancel.**
  A heartbeat answer with `cancel` set: SIGTERM, 30 seconds, SIGKILL, then the job is completed.
- **Lost lease.**
  A heartbeat answer of `lease_lost` or `job_not_claimed` (HTTP 409 and 404), or of `runner_unknown` (the runner was revoked): the same stop, then the directory is deleted and nothing is completed.
- **After the process exits** the launcher's process group is sent SIGKILL. The launcher always starts its container in a session of its own, outside that group, so the group signal does not reach a container; the container is ended by the launcher (it forwards SIGTERM and kills it after its own ten seconds) or, under Docker and Podman, by the runner's container removal.
- **When supervising itself fails** (any exception in the runner), the runner sends SIGTERM first, waits as long as the launcher's own grace for its container (ten seconds plus a margin), and only then SIGKILL to the launcher's group: a SIGKILL alone would orphan the container.

### When the runner dies

What is guaranteed, and what is not:

- On Linux the launcher child is started with `prctl(PR_SET_PDEATHSIG, SIGTERM)` (through `ctypes`), so the kernel sends it SIGTERM when the runner's thread that started it dies, SIGKILL of the runner included; the launcher forwards SIGTERM to its container.
  Elsewhere the runner logs `pdeathsig_unavailable` for each job and has no such guarantee.
- Each launcher's pid, process-group id and start time (field 22 of `/proc/<pid>/stat`, the `ps` start time elsewhere) go into `<root>/pids/<job id>.json` before the heartbeat starts and are deleted when the job ends.
  When the runner starts it sends SIGTERM, and 30 seconds later SIGKILL, to every recorded group whose leader still has the recorded start time (a reused pid is left alone); then it removes this runner's labeled containers; only then it deletes the leftover job directories and pid files.
- SIGHUP is handled exactly as SIGTERM.
- After a normal end, a cancel, a lost lease or a deadline kill, the launcher's process group is sent SIGKILL (not the container, which is outside it: see above).

What the launcher adds, for its own death (a SIGKILL, a crash, the runner's SIGKILL of its group): on Linux it starts its container child with `prctl(PR_SET_PDEATHSIG)`, SIGKILL under Apptainer, so that the starter dies with the launcher, and SIGTERM under Docker and Podman, so that the engine's client passes it to the container.
It also stops its container itself at the job's end time, without waiting for a signal, and records `time_limit` ([the launcher](launcher.md#the-container)).
Where the launcher cannot ask for the signal it logs `pdeathsig_unavailable` once and has no such guarantee.
This repository's tests prove that the child is sent the signal when the launcher is killed, with the stand-in runtimes; what a real Apptainer or Docker then does with it is checked in the staging drills.

What is not guaranteed:
the kernel clears the parent-death signal on exec of any set-user-ID, set-group-ID or file-capability binary, so such a binary anywhere in the exec chain between the launcher and the container ends the guarantee for that container: a setuid-mode Apptainer install is one case (its starter is a set-user-ID binary), and an engine packaged with a set-user-ID helper, a snap-packaged one for example, is another; an Apptainer container is then orphaned as before, forever, since no runner removes it, and an engine's container stays only until the runner's container removal at the end of the job or at its next start, which bounds that case;
under Docker and Podman the signal is a SIGTERM that nothing follows with a SIGKILL once the launcher is gone, and a client killed with SIGKILL passes nothing on at all, so an entry that is the first process of its container (process identifier 1) with no handler for SIGTERM stays until the runner removes the container, at the end of the job or at the runner's next start;
a process that calls `setsid` leaves the launcher's group and is not reached by a group signal, though one that does so inside the container's namespace is still bounded by the container;
and the runner does not make itself a child subreaper (`PR_SET_CHILD_SUBREAPER`) to hunt the descendants a job leaves behind: that is not done, and no card tracks it yet.

### Containers (Docker and Podman)

The launcher labels each container `nemar.job=<job id>` and `nemar.runner=<runner id>` (the id the runner passes as `NEMAR_RUNNER_LABEL`).
When a job ends, however it ends (a lost lease, a cancel, a deadline kill, any exit of the launcher, or an error in the runner), the runner runs `<engine> ps -aq --filter label=nemar.runner=<runner id> --filter label=nemar.job=<job id>` and then `<engine> rm --force --volumes <ids>` when any id is listed.
When the runner starts it does the same with `--filter label=nemar.runner=<runner id> --filter label=nemar.job`, which removes the containers a dead run of this runner left and never another runner's, on a daemon the runners share.
Each removal is logged as `container_removed` with the job id (none, for the start-up sweep) and the short container id.
Without the registration id nothing is removed.
Ids the engine prints are checked to be hexadecimal before they reach `rm`; a failing engine is logged (`container_cleanup_failed`) and never stops the job's completion.
Under Apptainer nothing is done.
Each container `ps` lists is inspected and removed only if its labels, read back, include every one asked for, so an engine that ORs repeated `--filter label=` cannot get another runner's container removed.
Run ONE runner process per registration: two processes on one registration (with different roots) share the label and remove each other's containers at start.
The launcher refuses Podman for now (`podman_unsupported`), so `doctor` answers no on both engine lines under it.

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

## Sharing a workstation

A workstation is often not dedicated to this runner: other people compute on it, their processor use is rare, their memory use can be large, and a deadline can make the graphics processing unit (GPU) busy for days.
The runner is built to be a good neighbor there.
The registration is still the ceiling (NEMAR never hands this runner more than it registered, and this section changes no route and no protocol field); what follows only makes the runner claim, and start, less.
Nothing preempts a job: a job that has started runs to its end, its deadline, a cancel or a revoke.

### Before each claim

The runner reads the host's live state before it asks for work, and sends no claim at all unless every rule holds, so a decline costs nothing and the Worker counts nothing.

| Rule | Idle means | Reason when it fails |
|---|---|---|
| free scratch | free space under `root` is at least `max_scratch_gb` (always checked, even with `enabled` false) | `scratch` |
| memory | `MemAvailable` less `reserve_memory_mb` is at least `claim_memory_mb` | `memory`, or `memory_unknown` when `/proc/meminfo` has no `MemAvailable` |
| load | the one-minute load average divided by the CPUs this process may use is at most `max_load_per_core` | `load`, or `load_unknown` |
| pause | no entry named `pause` is in any reservation directory | `paused` |
| backoff | no job was given up for lack of room less than `backoff_seconds` ago | `backoff` |

A reading that cannot be taken counts as busy.
The readings are Linux's: `/proc/meminfo` and `/proc/loadavg` (the load comes from `os.getloadavg()` where there is no `/proc/loadavg`), the CPU count from the process's affinity mask where there is one.
Where there is no `/proc/meminfo` (macOS) the host is always busy for memory unless `enabled` is false, so a workstation runner is a Linux one.
`claim_memory_mb` is the largest memory a claim can hand this runner: set it to the memory the runner was registered with.
The reserve is the memory always left to people, the larger of 8 GiB and a tenth of the host's memory, unless set.

A decline is logged as `claim_skipped` with `reason` and the numbers it was judged on (`free_gb`, `mem_available_mb`, `mem_total_mb`, `reserve_mb`, `need_mb`, `load_per_core`, `max_load_per_core`, `cpus`), once each time the reason changes and not on every poll; the first claim that follows is `claim_resumed` with `after` the last reason.
The runner then polls again after the usual 15 seconds.

### Just before a claimed job starts

The same rules, except the backoff (which is about claiming), are asked again after the image step and before the launcher starts, for the job's own memory (`default_job_memory_mb` for a job the claim gave none under Docker or Podman, `claim_memory_mb` under Apptainer).
While they fail the runner asks again every 15 seconds (`start_waiting` once per reason, `start_resumed` when there is room) until 540 seconds after the claim, the lease less a minute, or the job's own deadline if that comes first.
Then no heartbeat has been sent and none will be, so the job is abandoned exactly as a pull that outlasted the lease is: nothing is completed, the lease expires, and the Worker releases the claim and counts it (a job has three).
The log line is `claim_abandoned` with the reason and the numbers, and the runner sets a backoff: no claim for `backoff_seconds` (half an hour), so one busy host never spends the same job's three claims in a row.
A job whose wait ends because the runner is stopping is refused as it is when a stop reaches it during a pull.

### Pausing it

A person who needs the machine creates an entry named `pause` in a reservation directory, whatever it is (`touch <reservation directory>/pause`), and removes it when done.
The runner only looks at the name with `lstat`: it never opens, reads or follows it, so a file, a link, a directory or a pipe of that name all pause it, and so does a reservation directory it cannot search, since a pause it cannot see is respected.
A running job is not stopped, a claimed job that has not started waits as above, and no new claim goes out.
Who can pause the runner is whoever can create a file in a reservation directory: make a shared one the way the lab decides (a directory the lab's group can write), and name it in `reservation_dirs`.
The default is the `reservations` directory beside the configuration file, which the runner never creates.
The card for the GPU reads the same directories.

### Priority

At start the runner raises its own niceness to 19 (it never lowers it) and puts its input and output (I/O) scheduling class in idle, so that every process it starts afterwards inherits both: the launcher, image pulls, and Apptainer and rootless Podman containers (`priority_set` in the log).
The I/O class is set by `ioprio_set` through `ctypes` on Linux for x86-64, arm64 and RISC-V, and by `ionice -c 3 -p <own pid>` from the launch PATH's trusted directories otherwise.
A part that cannot be applied is logged as `priority_unavailable` and the runner goes on.
A container the Docker daemon starts is not the runner's descendant, so neither reaches it: the launcher gives that container the lowest CPU weight, `--cpu-shares 2`, when the engine can apply it ([the launcher](launcher.md#what-differs-from-the-apptainer-path)).
A container's hard `--cpus` and `--memory` limits and its `--oom-score-adj=1000` stay, so under memory pressure the kernel ends the job before a person's process.

### Configuration

An optional `availability` object in the configuration file.
Without it these defaults apply, so a configuration written before it existed keeps working, now with the live check; a key left out has its default, and an unknown key is refused as the top level's are.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | `false` turns off the memory, load, pause and backoff rules and the check before a job starts; the free-scratch rule stays |
| `max_load_per_core` | `0.9` | the most load per CPU that still leaves room (a number above 0 and at most 1024) |
| `reserve_memory_mb` | the larger of 8192 and a tenth of the host's memory | memory always left to people, megabytes |
| `claim_memory_mb` | `default_job_memory_mb` when set, else `4096` | the memory a claim may hand this runner, megabytes; set it to the registered memory |
| `backoff_seconds` | `1800` | how long claims stop after a job was given up |
| `reservation_dirs` | the `reservations` directory beside the configuration file | a list of at most 16 absolute paths searched for `pause` |

For example, to leave 16 GiB to people on a 128 GiB workstation registered for jobs of up to 32 GiB, with a lab-wide pause directory:

```json
"availability": {
  "reserve_memory_mb": 16384,
  "claim_memory_mb": 32768,
  "reservation_dirs": ["/path/to/shared/reservations"]
}
```

### What it prints

`python3 -m nemar_runner availability` (with `--config <file>` before or after the command) prints one line and exits 0 when the host has room for a claim now and 1 when it does not:
`availability: idle` or `availability: busy reason=<word>`, then the numbers it was judged on, never a path or a secret.
It cannot know a backoff, which lives in a running runner.
`doctor` adds two informational lines, `availability` (`yes`, or `no (reason=<word>)`) and `priority` (`yes` when a process of its own can take niceness 19 and the idle I/O class on this host), and neither changes the exit status.

## Logs

JSON lines, one object per line, to `<root>/runner.log` (mode 0600) and to standard error.
A line holds job ids, counts, fixed words and exception class names: never the secret, a token, a lease, a presigned address, a launch value, or anything the job or the launcher wrote.
Lines go through a bounded queue to a writer thread, so a standard error or file that blocks drops lines (counted) and never stalls the supervisor.
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
| `engine limits enforceable`, `engine mode accepted` (Docker and Podman only; asked only when `launcher`, `launcher_dir trusted` and `launcher python` are yes, because the preflight imports the launcher's code) | the two answers of the launcher's own engine preflight (`nemar_launch.runtime.engine_preflight(engine_path, engine_name)`, called in a child process run as the launcher is), so `doctor` and the launcher cannot disagree about an engine; rootless Docker without enforceable limits answers no |
| `availability` (informational) | the host has room for a claim now ([Sharing a workstation](#sharing-a-workstation)); a no names the reason and is not a failure |
| `priority` (informational) | a process of its own can take niceness 19 and the idle I/O class on this host |

Exit 0 only if every line but the Apptainer `limits` line and the two informational ones is yes: a host that cannot enforce limits still runs jobs, with the watchdog, and a host that is busy right now or cannot lower its priority still runs them.
`run` and `once` run the same probe the first time an image is available and set `NEMAR_ENFORCE_LIMITS` from it; Docker and Podman are never claimed, because the launcher owns their flags and checks them with the preflight.

## Developing

```bash
uv run --no-project --python 3.9 --with pytest --with pytest-timeout pytest tests/runner -q     # the runner on its floor (each test is limited to two minutes)
uv run pytest tests/contract/test_runner_drift.py -q                      # against the pinned nemar-cli commit
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run ty check --python-version 3.9 launcher observer runner
```

The tests run real child processes and real HTTP; NEMAR's Worker is a recording stand-in that answers fixtures and checks each request body against the pinned schema, and the container runtime is a stand-in executable where a test is about how it is called.
The one exception is `tests/e2e/test_runner_stack.py`, described next, which runs against NEMAR's real routes and a real container runtime.
The runner against NEMAR's real routes is `tests/e2e/test_runner_stack.py` (marker `runner_stack`): it starts nemar-cli's local stack (`backend/scripts/compute-local-stack.ts serve --seed-fixture`) from a full checkout named by `NEMAR_CLI_DIR` (at the pinned commit, with `bun install --frozen-lockfile --ignore-scripts` run in its root and in `backend/`), registers a runner with the stored owner key, dispatches `nemar-floor-zarr` with `executor: "runner"` with the stored administrator key, runs `once`, and asserts what the Worker's own job record says.
One case takes the job to `finished` in a real Apptainer container (the tiny image of `tests/images/tiny-recipe`, built into `<NEMAR_E2E_SIF_DIR>/<image id>.sif`, default `.cache/runner-stack-sif`); the other, with an empty `sif_root`, ends `failed` with `image_missing` before any container starts.
Without Apptainer the first skips with "apptainer absent"; with `NEMAR_REQUIRE_APPTAINER=1` (the `runner-stack` CI job) a skip is a failure.
The variable keeps the name of the Apptainer gate of the other tests but gates every way this module can fail to run: a missing Apptainer, SIF, bun, nemar-cli checkout or checkout dependency is a failure under it, never a skip.
The runner is given a private launch PATH holding `python3` and an `apptainer` wrapper that executes the real binary: the runner's trust check vets the wrapper, not the real Apptainer binary or its directory, and the default launch PATH is not exercised by this test.
The other jobs deselect the marker.
The drift test needs a nemar-cli checkout that has the pinned commit (`NEMAR_CLI_RUNNER_DIR`, or a `nemar-cli` directory beside this checkout), and fails instead of skipping when `NEMAR_REQUIRE_RUNNER_DRIFT=1`.
