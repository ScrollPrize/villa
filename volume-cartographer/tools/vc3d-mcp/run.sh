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

# A venv is healthy only if its python exists AND can import the runtime deps.
# A partial venv (python but no pip, or pip but no mcp) is rebuilt from scratch.
needs_setup=0
if [ ! -x "$VENV/bin/python" ]; then
    needs_setup=1
elif ! "$VENV/bin/python" -c "import mcp, vc3d_mcp" >/dev/null 2>&1; then
    echo "[vc3d-mcp] Existing $VENV is incomplete; rebuilding ..." >&2
    rm -rf "$VENV"
    needs_setup=1
fi

if [ "$needs_setup" -eq 1 ]; then
    echo "[vc3d-mcp] Setting up Python environment in $VENV ..." >&2
    PYTHON_BIN="${PYTHON:-python3}"
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "[vc3d-mcp] No '$PYTHON_BIN' found on PATH. Install Python 3.10+ or set PYTHON=/path/to/python3." >&2
        exit 1
    fi
    "$PYTHON_BIN" -m venv "$VENV" >&2
    "$VENV/bin/pip" install -q -e "$DIR" >&2
    echo "[vc3d-mcp] Setup complete." >&2
fi

exec "$VENV/bin/python" -m vc3d_mcp "$@"
