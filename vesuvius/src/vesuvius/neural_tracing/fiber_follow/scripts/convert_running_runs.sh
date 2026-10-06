#!/usr/bin/env bash
# One-off: convert the latest checkpoint of the three runs trained before the model cleanup, beside each source:
#   CHECKPOINT.pt -> CHECKPOINT.converted.pt, plus RUN/run.converted.json
# Run it after stopping a run (on the machine holding its checkpoints); pass checkpoints to convert others:
#   scripts/convert_running_runs.sh [CHECKPOINT.pt ...]
# Then resume with: python train/train.py --config RUN/run.converted.json --resume CHECKPOINT.converted.pt
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
VES="$(cd "$FF/../../../.." && pwd)"
PYTHON=${PYTHON:-$VES/.venv/bin/python}
export PYTHONPATH="$VES/src${PYTHONPATH:+:$PYTHONPATH}"
latest() {  # last.pt, else the highest-numbered ckpt_*.pt
    if [[ -f "$1/last.pt" ]]; then echo "$1/last.pt"; else ls "$1"/ckpt_*.pt | sort | tail -1; fi
}
if [[ $# -eq 0 ]]; then
    set -- "$(latest "$FF/output/mixed_ct_afv_unified_v1")" \
           "$(latest "$FF/output/unified_flow_20261006/runs/unified_flow_v6_adalnzero")" \
           "$(latest "$FF/output/sequence_v1")"
fi
for source in "$@"; do
    "$PYTHON" "$FF/scripts/convert_checkpoint.py" "$source" "${source%.pt}.converted.pt" \
        --run-config "$(dirname "$source")/run.converted.json"
done
