#!/usr/bin/env bash
# Full spatial history and query-dependent retrieval; starts a separate run.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
export RUN_NAME=${RUN_NAME:-axial_spatial_memory_run1}
exec bash "$FF/scripts/launch_unified.sh" \
    --memory-version 4 --spatial-recent 2 --spatial-archive 8 --spatial-retrieve 2 \
    --trajectory-window 4 "$@"
