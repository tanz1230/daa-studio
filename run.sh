#!/usr/bin/env bash
# DAA Studio launcher (Linux / macOS). Opens the app in your browser.
#   ./run.sh                                  start screen: create or open a project
#   ./run.sh --project PATH                   straight into a project
#   ./run.sh --host 0.0.0.0 --port 8765       let annotators on other machines connect
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"

if ! "$PY" -c "import numpy, scipy, yaml, cv2" 2>/dev/null; then
  echo "DAA Studio needs numpy, scipy, pyyaml and opencv. Install them with:"
  echo "    $PY -m pip install -r \"$HERE/requirements.txt\""
  exit 1
fi
cd "$HERE"
exec "$PY" -m studio serve "$@"
