#!/bin/sh
cd "$(dirname "$0")" || exit 1
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(not ((3,11) <= sys.version_info[:2] <= (3,12)))' 2>/dev/null; then
    "$candidate" bootstrap.py "$@"
    result=$?
    if [ "$result" -ne 0 ]; then
      printf '\nPress Return to close. '
      read -r response
    fi
    exit "$result"
  fi
done
printf 'Install Python 3.11 or 3.12 from https://www.python.org/downloads/ with Tcl/Tk, then open this launcher again.\n'
printf 'Press Return to close. '
read -r response
exit 1
