#!/usr/bin/env bash
# macOS / Linux launcher — use run_camera.py for full cross-platform support.
# This script delegates to run_camera.py using whichever Python is active.

cd "$(dirname "$0")"

# Find python: honour venv/conda if active, else fall back to python3
if command -v python &>/dev/null; then
    PYTHON="python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "[ERROR] Python not found. Install Python 3.9+ or activate your conda/venv."
    exit 1
fi

exec $PYTHON run_camera.py "$@"
