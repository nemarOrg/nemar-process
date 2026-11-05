# The Tapis definitions and the observer: what the operator does

NEMAR runs its compute jobs through one Tapis tenant, on one execution system, as one community account.
This page is for the person who sets that up and keeps it running: what the repository provides, what only the operator can supply, and how the definitions are registered.
Nothing here is registered by CI or by a coding agent, and no tenant, host, account or allocation is named anywhere in this repository.
Real values live in secrets and in the gated documentation, never here.

Abbreviations used here: application programming interface (API), JavaScript Object Notation (JSON), Secure Shell (SSH), Secure Hash Algorithm 256 (SHA-256), Singularity Image Format (SIF), continuous integration (CI), Message Passing Interface (MPI), Simple Storage Service (S3).

## What is here

| Path | What |
|---|---|
| `deploy/tapis/apps/nemar-launch.json` | the launcher application: ZIP runtime, `BATCH` jobs |
| `deploy/tapis/apps/nemar-observer.json` | the queue observer application: ZIP runtime, `FORK` jobs |
| `deploy/tapis/systems/exec.json.tmpl` | the execution system template: SSH, Slurm, nine logical queues |
| `deploy/tapis/example.env` | an example value for every placeholder; CI renders every definition with it |
| `deploy/tapis/openapi/` | Tapis's own Applications and Systems specifications, vendored byte for byte (`SOURCE` says from where; `LICENSE-tapis` is their licence) |
| `deploy/tapis/schemas/` | the JSON Schemas derived from those specifications |
| `scripts/derive_tapis_schemas.py` | derives the schemas; `--check` fails on any difference |
| `scripts/register.py` | renders the definitions and registers what is absent; a dry run unless told otherwise |
| `scripts/build_observer_zip.py` | builds the observer zip, byte for byte the same every time |
| `observer/` | `observe.py`, the observer program, and `tapisjob_app.sh`, the entry Tapis runs |

## One tenant, one owner, one account

There is a single NEMAR Tapis tenant, and NEMAR owns every job in it.
No person has a Tapis user: people reach compute through NEMAR's API, and NEMAR submits on their behalf.
Every job on the execution system runs as one static host account, the community account, which is what `effectiveUserId` fixes.

So no end user needs, or may be given, Tapis access of any kind.
Do not share the system or either application with anyone but NEMAR's own service identity, and do not grant anyone permissions on them.
`scripts/register.py` never calls the sharing or permission routes, and a test pins that.

## What the operator supplies

Every operator-specific value is a placeholder, written `${NAME}` in a definition.
Values come from the environment or from a file passed as `--vars FILE` (`NAME=value` lines, raw, no quoting, no newline in a value; the file wins over the environment).
An unresolved placeholder is an error, and so is a value of the wrong shape.
`deploy/tapis/example.env` has one example value for each; copy it outside the repository and fill in the real ones.

| Placeholder | Used in | What it is |
|---|---|---|
| `EXEC_HOST` | system | the host Tapis connects to over SSH, with no scheme and no path |
| `COMMUNITY_ACCOUNT` | system | the one host account every job runs as |
| `AUTHN_METHOD` | system | how Tapis authenticates to the host: `PKI_KEYS`, `TMS_KEYS`, `CERT` or `PASSWORD` |
| `EXEC_ROOT_DIR` | system | a directory only the community account can read (mode 0700) on the cluster's scratch filesystem, never a home directory: Tapis's job directories are created under it |
| `SIF_ROOT` | launcher | the directory holding the image files, each named `<64 hex digits of the image digest>.sif` ([images.md](images.md)) |
| `LOCAL_DATASETS_ROOT` | launcher | the root of NEMAR's own copy of datasets on the node, `<root>/<dataset id>/<version>/`; a directory that does not exist means no local copy, and the launcher downloads |
| `NODE_LOCAL_TMPDIR` | launcher | a node-local directory for scratch; the launcher makes its own private directory inside it |
| `SLURM_ACCOUNT_CPU`, `SLURM_ACCOUNT_GPU` | system, observer | the scheduler accounts charged for the CPU partitions and for the GPU partitions (the same name twice if one account holds both) |

Credentials are not placeholders.
A placeholder named like a credential (it holds `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `PRIVATE` or `KEY`) is refused, so a definition can never take one.

## Before registering

1. **Create the directories.**
   Create `EXEC_ROOT_DIR` on scratch with mode 0700, owned by the community account.
   The system's `jobWorkingDir` is `<root>/jobs`, so Tapis creates `<root>/jobs/exec/<job>`, `<root>/jobs/input/<job>` and `<root>/jobs/output/<job>`: the execution directory (where the zip is unpacked and where Tapis writes its launch files) is never the output directory (what is archived), and neither is in the home directory, whose quota these jobs would otherwise eat.
   The system's `rootDir` is `/`, on purpose: Tapis creates a job's directories at `rootDir` plus the path, but the variables it gives the job (`_tapisExecSystemOutputDir`) hold the rootDir-relative path, and the launcher reads that variable as a host path, so the two agree only when `rootDir` is `/`.
   The root's mode 0700 is what keeps other accounts out; Tapis's `rootDir` is not used for that.
   Create `SIF_ROOT` and install the images into it; only the community account may write there, because the launcher trusts a file's name and does not hash it.
2. **Check the host.**
   The launcher and the observer need Python 3.9 or newer outside the container.
   If the host's `python3` is older, add a `FIXED` variable `NEMAR_HOST_PYTHON` naming a newer interpreter to both applications and register a new version; it is deliberately not offered to job requests.
3. **Publish the zips.**
   Both definitions name a GitHub release asset: `nemar-launch-<version>.zip` under the tag `nemar-launch-v<version>`, and `nemar-observer-<version>.zip` under `nemar-observer-v<version>`, in this repository.
   Build them with `scripts/build_launcher_zip.py` and `scripts/build_observer_zip.py`; the builds are deterministic, and the SHA-256 recorded in each definition's `notes` (`launcher_sha256`, `observer_sha256`) is the digest of that exact file.
   CI rebuilds both and fails if a recorded digest is wrong.
4. **Register the credential for the system, separately.**
   The execution system needs a credential (an SSH key, a Trust Manager setup or whatever the host's policy allows) that Tapis uses to reach the host as the community account.
   It is registered by the operator, through Tapis's own credential route and the operator's own tooling.
   This repository has no file, template, field or script for it, and `register.py` sends none: the system template has no `authnCredential`, and a test fails if one appears.

## Registering

```bash
export TAPIS_BASE_URL=<tenant-base-url>      # https, from the environment only
export TAPIS_TOKEN=<a token allowed to create systems and applications>  # from the environment only; never printed

uv run python scripts/register.py --vars /path/to/values.env system deploy/tapis/systems/exec.json.tmpl
uv run python scripts/register.py --vars /path/to/values.env app deploy/tapis/apps/nemar-launch.json deploy/tapis/apps/nemar-observer.json
# read what it would send, then the same commands with --apply
```

A dry run (the default) renders each definition, checks it against the derived schema, and prints the method, the path and the hash of the body it would send.
It sends nothing, not even a read, and needs neither variable.

`--apply` reads each definition first and creates it only if it is absent.
A second run therefore does nothing.
An existing version is left exactly as it is: a registered application version is never modified, and both applications are created `locked`.
To change an application, publish a new version, which follows the launcher's or the observer's version (versions are the release process's, never edited by hand).
A 409 on the create means the definition appeared after the read; that is reported as an error, because the run did not look at what is there.
The system is registered before the applications, because they name it.

The token is in `X-Tapis-Token` only, is in no output, and no redirect is followed.
Nothing about the base address is printed either, because it names the tenant.

## Writing a definition

Tapis's own macros, `${JobWorkingDir}` and `${JobUUID}`, share the placeholder syntax.
A definition writes them with a doubled dollar sign, `$${JobUUID}`, and they come out as `${JobUUID}`.
A macro written without the doubling is an unresolved placeholder and an error.

The render works on the parsed JSON, never on its text, so a value can fill a placeholder and can never change the structure of the document.
After rendering, the document must validate against `deploy/tapis/schemas/`, which is Tapis's own specification made strict: every object that declares properties refuses any other key, so a misspelled key fails here and not at the tenant.
The one object left open is `notes`, which Tapis defines as free-form metadata.

To move to newer Tapis specifications, copy the two files from the new commit, update `deploy/tapis/openapi/SOURCE` with the new commits and SHA-256 sums, run `uv run python scripts/derive_tapis_schemas.py`, and commit the regenerated schemas with them.
The derivation refuses a keyword it has not been taught, so a newer specification cannot weaken a schema unnoticed.

## The rules the definitions keep

- **Launch files are never archived, in two layers.**
  Tapis writes the job's launch files, `tapisjob.env` and `tapisjob.sh`, and archives them with the output unless told not to; `tapisjob.env` holds the launch specification, which holds the job's upload token (the observer's token too).
  The Tapis field is `jobAttributes.parameterSet.archiveFilter.includeLaunchFiles`.
  Both applications set it to `false` and also list both files in `excludes`; NEMAR's Tapis client sets the same field to `false` on every job it submits.
  `tests/deploy/test_tapis_client_agreement.py` pins that the two sides agree, against the client's own source.
  After registering, read the application back and look at that field; after the first job, list its archive and look for the two files.
- **The output directory is per job and is not the execution directory.**
  Both are under the job working directory and keyed by `${JobUUID}`, and neither is under a home directory.
  `scripts/tapis_policy.py` computes the directory the way Tapis does and refuses a system whose `rootDir` is not `/` or whose working directory is not under the operator's scratch root.
- **A variable a submitter must not change is `FIXED`.**
  `NEMAR_SIF_ROOT`, `NEMAR_LOCAL_DATASETS_ROOT` and `TMPDIR` are fixed in the launcher application, and `NEMAR_SLURM_ACCOUNT` in the observer's.
  The launch specification, the observer's URL and its token are `REQUIRED`: the job request supplies them.
  The scheduler's `CUDA_VISIBLE_DEVICES` is not defined at all, because a value in a definition would hide the one the scheduler sets for each job; the launcher passes the scheduler's value into the container for a GPU job when it is plain.
- **No archive system, no storage credential.**
  There are no archive system fields in either application and no S3 system anywhere: results reach NEMAR through the launcher's own upload, so Tapis archives only to its default place on the execution system, with the launch files left out.
- **Nothing runs around the launcher.**
  No command prefix, no MPI, no file inputs (`strictFileInputs` is on), no container arguments.

## The observer

`nemar-observer` is a `FORK` application that NEMAR submits every 15 minutes in `quick` mode and once a day in `daily` mode.
It runs `squeue`, `sinfo` and `expanse-client` on the execution system's login node (and `sacct` in `daily` mode), summarises them, and posts one `nemar.observation.v1` document to NEMAR with a token valid for that one run.
The program's own header (`observer/observe.py`) documents the formats it reads, its exit statuses and what it never does, and it takes its settings from the variables below.

| Variable | Mode in the definition | Meaning |
|---|---|---|
| `NEMAR_OBSERVER_URL` | `REQUIRED` | the NEMAR route to post to (https) |
| `NEMAR_OBSERVER_TOKEN` | `REQUIRED` | the bearer token, one run only |
| `NEMAR_OBSERVE_MODE` | `INCLUDE_BY_DEFAULT`, `quick` | `quick` or `daily` |
| `NEMAR_OBSERVE_JOBIDS` | `INCLUDE_ON_DEMAND` | Slurm job ids to ask expected start times for, at most 500 |
| `NEMAR_OBSERVE_DAY` | `INCLUDE_ON_DEMAND` | the day of the accounting export, default yesterday in UTC |
| `NEMAR_SLURM_ACCOUNT` | `FIXED` | the accounts the export is for, from the two account placeholders |

What it promises: it never invents (a command that fails, is missing or times out is reported as that, and what it would have said is `null`, never zero); it never prints the token, never gives it to a child process and never follows a redirect; it posts at most 1 MiB, cutting the daily accounting rows from the end and saying so; and it tries three times, only for a failure that may clear.
The accounting export carries a job's id, partition, state, elapsed time, allocated resources and times, and no user names.

The NEMAR route it posts to is built in the gateway epic and does not exist yet.
If the site does not allow a `FORK` application to run these commands on a login node, the observer cannot run as designed; that is question 6 of the design page, and an escalation, not something to work around.

## Housekeeping

Tapis does not remove a job's execution, input or output directory, so they accumulate under `EXEC_ROOT_DIR`, and the observer adds one every 15 minutes.
The launcher already uploads what matters, so the archive is for diagnosis only: purge old job directories on a schedule the site's scratch policy allows.
The per-run token of an observer job sits in the execution directory's `tapisjob.env` until then; the root is mode 0700, and the token is valid for one run.

## What is unverified

None of this has run against a real tenant or a real node.
[unverified.md](unverified.md) lists what the definitions assume, in the order it would bite, and which of those are questions for the site operator.
If an operator's test contradicts the Tapis documentation, stop and tell the repository owner.
