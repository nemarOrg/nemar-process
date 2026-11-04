#!/bin/sh
# nemar-launch: the entry point Tapis runs for a ZIP-runtime application, on the compute node,
# outside the container. It finds a usable host Python (3.9 or newer), runs the launcher in
# isolated mode, and leaves the launcher's exit number in ${_tapisExecSystemOutputDir}/tapisjob.exitcode
# and as its own exit status. Exit numbers: 0 ok; 10 code_fetch_failed; 11 dataset_fetch_failed;
# 12 entry_not_found; 13 runtime_error; 14 out_of_memory; 15 time_limit; 16 result_upload_failed.
#
# POSIX sh and shell builtins only: no external command is needed to start (PATH may be minimal).
# It never echoes the environment: NEMAR_LAUNCH_SPEC holds the job token.
#
# NEMAR_HOST_PYTHON overrides the interpreter (it must itself be 3.9 or newer).

here=${0%/*}
[ "$here" = "$0" ] && here=.
here=$(cd "$here" && pwd -P)

out=${_tapisExecSystemOutputDir:-}
if [ -z "$out" ]; then
  echo "nemar-launch: _tapisExecSystemOutputDir is not set" >&2
  exit 13
fi

finish() {
  mkdir -p "$out" 2>/dev/null
  printf '%s\n' "$1" > "$out/tapisjob.exitcode"
  exit "$1"
}

usable() {
  "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1
}

py=
if [ -n "${NEMAR_HOST_PYTHON:-}" ]; then
  if usable "$NEMAR_HOST_PYTHON"; then
    py=$NEMAR_HOST_PYTHON
  fi
else
  for candidate in python3.12 python3.11 python3.10 python3.9 python3; do
    found=$(command -v "$candidate" 2>/dev/null) || continue
    if usable "$found"; then
      py=$found
      break
    fi
  done
fi

if [ -z "$py" ]; then
  echo "nemar-launch: no Python 3.9 or newer found (set NEMAR_HOST_PYTHON to one)" >&2
  finish 13
fi

# `python -I -m nemar_launch`, with the package directory named explicitly: in isolated mode (-I)
# Python ignores PYTHONPATH and the working directory, so `-m` alone cannot find a package that is
# only next to this script. runpy.run_module is what `-m` runs.
boot='import runpy, sys; sys.path.insert(0, sys.argv[1]); runpy.run_module("nemar_launch", run_name="__main__", alter_sys=True)'

# A SIGTERM to this script (Slurm's, at the time limit or on cancel) goes on to the launcher, which
# passes it to the container; this script then still writes tapisjob.exitcode.
child=
on_term() {
  if [ -n "$child" ]; then
    kill -TERM "$child" 2>/dev/null
  fi
}
trap on_term TERM

"$py" -I -c "$boot" "$here" &
child=$!
wait "$child"
rc=$?
while kill -0 "$child" 2>/dev/null; do
  wait "$child"
  rc=$?
done

finish "$rc"
