#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${FIBER_PYTHON:-$task_root/../../../../.venv/bin/python}"
task_run="${STEM_RUN_NAME:-mixed_ct_afv_stem32_run2}"
cd "$task_root"
# Use the migrated checkpoint's complete settings, then apply explicit overrides.
exec "$task_python" - "$task_root/output/$task_run/migration.json" "$@" <<'PY'
import json
import os
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
command = json.loads(manifest.read_text())['command']
command[0] = sys.executable
command += ['--resume', str(manifest.parent/'last.pt'), *sys.argv[2:]]
os.execv(command[0], command)
PY
