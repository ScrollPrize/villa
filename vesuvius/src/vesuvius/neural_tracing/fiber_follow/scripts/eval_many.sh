#!/bin/bash
# Evaluate several checkpoints on the full held-out seed set, N at a time.
#   scripts/eval_many.sh PARALLEL TAG=CKPT [TAG=CKPT ...]   (CKPT may be "field")
FF="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; O=$FF/output; VES="$(cd "$FF/../../../.." && pwd)"
par=$1; shift
cd "$VES"; export LD_LIBRARY_PATH=$VES/.venv/lib/python3.11/site-packages
for job in "$@"; do
    tag=${job%%=*}; ck=${job#*=}
    while [ $(jobs -rp | wc -l) -ge "$par" ]; do sleep 5; done
    ( .venv/bin/python "$FF/scripts/eval_ckpt.py" "$ck" --tag "$tag" 2>&1 | tail -1 > "$O/eval/eval_$tag.log" ) &
done
wait
