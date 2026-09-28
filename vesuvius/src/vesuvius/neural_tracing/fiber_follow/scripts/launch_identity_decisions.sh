#!/usr/bin/env bash
# Full identity-decision experiment with observed persistent seed references.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
export RUN_NAME=${RUN_NAME:-direct_identity_decisions_run1}
exec bash "$FF/scripts/launch_identity_shared_bank.sh" \
    --persistent-seed --decision-fraction 0.25 --candidate-weight 1.0 \
    --bank-wrong-continuation-probability 0 --anchor-prob 1.0 "$@"
