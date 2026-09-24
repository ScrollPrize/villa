#!/bin/bash
# Run training runs back to back, each followed by the 200-seed eval.
#   scripts/queue.sh STEPS NAME [NAME ...]   (args per run from output/logs/NAME.args)
FF="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
O=$FF/output; VES="$(cd "$FF/../../../.." && pwd)"
steps=$1; shift
for name in "$@"; do
    ck=$(printf "%s/%s/ckpt_%06d.pt" "$O" "$name" "$steps")
    "$FF/scripts/launch.sh" "$name" --steps "$steps" $(cat "$O/logs/$name.args")
    until [ -f "$ck" ]; do sleep 15; done
    sleep 20; "$FF/scripts/stop.sh" "$name"
    (cd "$VES" && LD_LIBRARY_PATH=$VES/.venv/lib/python3.11/site-packages .venv/bin/python \
        "$FF/scripts/eval_ckpt.py" "$ck" --tag "$name" 2>&1 | tail -1 > "$O/eval/eval_$name.log")
done
