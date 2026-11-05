#!/bin/sh
# nemar-observe: the entry point Tapis runs for the observer (a ZIP-runtime, FORK application), on
# the execution system's login node. It finds a usable host Python (3.9 or newer), runs lib/observe.py
# in isolated mode (no PYTHON* variables, no user site, no site module) with the same arguments, and
# leaves the program's exit status in ${_tapisExecSystemOutputDir}/tapisjob.exitcode and as its own.
#
# WHY THE FILE. For a FORK job Tapis checks that the process is gone and then reads
# <output dir>/tapisjob.exitcode; a missing file counts as success (from the Tapis source, to
# confirm on the tenant). A failed post would then be reported FINISHED, so this script writes the
# file itself, as the launcher's script does. The variable holds the rootDir-relative path, which is
# the host path because the system's rootDir is / (docs/operator.md).
#
# POSIX sh and shell builtins only. It never echoes the environment: NEMAR_OBSERVER_TOKEN is in it.
#
# NEMAR_HOST_PYTHON overrides the interpreter (it must itself be 3.9 or newer).
#
# Layout (scripts/build_observer_zip.py): this script at the root, the program in lib/.
umask 077

here=${0%/*}
[ "$here" = "$0" ] && here=.
here=$(cd "$here" && pwd -P)

out=${_tapisExecSystemOutputDir:-}
if [ -z "$out" ]; then
  echo "nemar-observe: _tapisExecSystemOutputDir is not set" >&2
  exit 13
fi

finish() {
  mkdir -p "$out" 2>/dev/null
  printf '%s\n' "$1" > "$out/tapisjob.exitcode"
  exit "$1"
}

usable() {
  "$1" -I -S -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1
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
  echo "nemar-observe: no Python 3.9 or newer found on this host" >&2
  finish 13
fi

"$py" -I -S "$here/lib/observe.py" "$@"
status=$?
finish "$status"
