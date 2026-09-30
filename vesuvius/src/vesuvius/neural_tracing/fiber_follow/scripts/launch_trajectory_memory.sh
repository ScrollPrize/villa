#!/usr/bin/env bash
# The regression launcher uses the live historical-slab model.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$FF/scripts/launch_memory.sh" "$@"
