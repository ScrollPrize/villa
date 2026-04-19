#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="title-tracer-0139"
WORKDIR="/home/ubuntu/villa-mesh/vesuvius-extend/vesuvius"
STATE_ROOT="/ephemeral/title_tracer_state"
LOG_ROOT="/ephemeral/title_tracer_logs"
LOCKFILE="${STATE_ROOT}/title_tracer_device4.lock"

required_env=(AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN)
for var_name in "${required_env[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required environment variable: ${var_name}" >&2
    exit 1
  fi
done

mkdir -p "${STATE_ROOT}" "${LOG_ROOT}"

if [[ -f "${LOCKFILE}" ]]; then
  lock_pid="$(cat "${LOCKFILE}" 2>/dev/null || true)"
  if [[ -n "${lock_pid}" ]] && kill -0 "${lock_pid}" 2>/dev/null; then
    echo "Refusing to start: lockfile is held by PID ${lock_pid}" >&2
    exit 1
  fi
  rm -f "${LOCKFILE}"
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux attach -t "${SESSION_NAME}"
  exit 0
fi

printf -v extra_args ' %q' "$@"
printf -v aws_key_q '%q' "${AWS_ACCESS_KEY_ID}"
printf -v aws_secret_q '%q' "${AWS_SECRET_ACCESS_KEY}"
printf -v aws_token_q '%q' "${AWS_SESSION_TOKEN}"

read -r -d '' command <<EOF || true
set -euo pipefail
echo \$\$ > "${LOCKFILE}"
trap 'rm -f "${LOCKFILE}"' EXIT
cd "${WORKDIR}"
export AWS_ACCESS_KEY_ID=${aws_key_q}
export AWS_SECRET_ACCESS_KEY=${aws_secret_q}
export AWS_SESSION_TOKEN=${aws_token_q}
export AWS_DEFAULT_REGION=us-east-1
export CUDA_VISIBLE_DEVICES=4
ts=\$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=src /home/ubuntu/villa-mesh/vesuvius/.venv/bin/python -m vesuvius.neural_tracing.autoreg_mesh.batch_extend_title_hunt${extra_args} 2>&1 | tee "${LOG_ROOT}/run_\${ts}.log"
EOF

tmux new-session -d -s "${SESSION_NAME}" "bash -lc $(printf '%q' "${command}")"
tmux attach -t "${SESSION_NAME}"
