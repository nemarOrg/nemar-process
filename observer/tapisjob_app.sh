#!/bin/sh
# nemar-observe: the entry point Tapis runs for the observer (a ZIP-runtime, FORK application), on
# the execution system's login node. It finds a usable host Python (3.9 or newer) and runs
# lib/observe.py in isolated mode (no PYTHON* variables, no user site, no site module), passing on
# its arguments and its exit status. For a FORK job Tapis reads the exit status itself.
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
  exit 13
fi

exec "$py" -I -S "$here/lib/observe.py" "$@"
