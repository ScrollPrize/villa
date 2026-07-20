#!/usr/bin/env bash
# Self-bootstrapping launcher for the VC3D Agent Bridge MCP server.
#
# Building VC3D via CMake only produces the C++ binary -- it does nothing to
# set up this directory's Python environment. Registering THIS script (not
# .venv/bin/python -m vc3d_mcp directly) with an MCP client means no manual
# `python3 -m venv` / `pip install` step is needed after building VC3D from
# source: the first time an MCP client launches this script, it creates the
# venv and installs the package; every launch after that is a plain exec
# with no setup overhead.
#
# Usage (same arguments as `python -m vc3d_mcp`):
#   ./run.sh --launch /path/to/build/bin/VC3D
#   ./run.sh --socket /path/to/socket
#   ./run.sh                                    # pure auto-discovery/auto-launch
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/.venv"

if [ ! -x "$VENV/bin/python" ]; then
    echo "[vc3d-mcp] First run: setting up Python environment in $VENV ..." >&2
    PYTHON_BIN="${PYTHON:-python3}"
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "[vc3d-mcp] No '$PYTHON_BIN' found on PATH. Install Python 3.10+ or set PYTHON=/path/to/python3." >&2
        exit 1
    fi
    "$PYTHON_BIN" -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q -e "$DIR"
    echo "[vc3d-mcp] Setup complete." >&2
fi

exec "$VENV/bin/python" -m vc3d_mcp "$@"
