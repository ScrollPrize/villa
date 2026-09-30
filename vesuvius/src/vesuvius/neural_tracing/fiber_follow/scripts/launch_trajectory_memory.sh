#!/usr/bin/env bash
# The regression launcher uses the sole observation-memory model.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$FF/scripts/launch_memory.sh" "$@"
