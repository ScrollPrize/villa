#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: vc-mcp-setup.sh <command>

Read-only:
  doctor       Check source, build, and optional runtime requirements.
  print-env    Print a reviewed shell environment template (no secrets).
  config       Print a generic Streamable HTTP client configuration as JSON.

Side effects:
  configure    Configure the MCP target (may fetch pinned CMake dependencies).
  build        Build vc_mcp_server and vc_grow_seg_from_seed.
  serve        Run the server over loopback Streamable HTTP. Requires
               VC_MCP_AUTH_TOKEN (source the reviewed print-env output).
  test         Build and run the core vc_mcp_* tests.
  install-python
               Create/update the optional analysis venv and install
               requirements-staging.txt. Requires explicit install approval.

Environment:
  VC_REPO_ROOT         Volume Cartographer root (auto-detected by default)
  VC_MCP_PRESET        Linux CMake preset (default: dev-clang)
  VC_MCP_BUILD_DIR     Explicit build directory
  VC_MCP_PYTHON_ENV    Analysis venv (default: ~/.cache/vc-mcp/venvs/analysis)
  AGENTS_AGENT_MODE=1  Marks an agent-run session
  AGENTS_ALLOW_INSTALL=1
                       Required with agent mode for install-python
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_root="$(cd "$script_dir/../../../.." && pwd)"
repo_root="${VC_REPO_ROOT:-$default_root}"

fail() { printf 'error: %s\n' "$*" >&2; exit 1; }
info() { printf '%s\n' "$*" >&2; }

require_repo() {
  [[ -f "$repo_root/CMakeLists.txt" ]] || fail "not a VC checkout: $repo_root"
  [[ -f "$repo_root/apps/VC3D/mcp/CMakeLists.txt" ]] || fail "MCP source missing under $repo_root"
}

preset="${VC_MCP_PRESET:-dev-clang}"
if [[ -n "${VC_MCP_BUILD_DIR:-}" ]]; then
  build_dir="$VC_MCP_BUILD_DIR"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  build_dir="$repo_root/build-macos"
else
  build_dir="$repo_root/build/$preset"
fi
python_env="${VC_MCP_PYTHON_ENV:-$HOME/.cache/vc-mcp/venvs/analysis}"
server="$build_dir/bin/vc_mcp_server"
grow="$build_dir/bin/vc_grow_seg_from_seed"

command_status() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    printf 'ok      %-18s %s\n' "$name" "$(command -v "$name")"
  else
    printf 'missing %-18s\n' "$name"
  fi
}

file_status() {
  local label="$1" path="$2"
  if [[ -e "$path" ]]; then
    printf 'ok      %-18s %s\n' "$label" "$path"
  else
    printf 'missing %-18s %s\n' "$label" "$path"
  fi
}

doctor() {
  require_repo
  printf 'repository: %s\n' "$repo_root"
  printf 'platform:   %s/%s\n' "$(uname -s)" "$(uname -m)"
  printf 'build dir:  %s\n\n' "$build_dir"
  command_status cmake
  command_status ninja
  command_status git
  command_status python3
  command_status uv
  command_status openssl
  if [[ "$(uname -s)" == "Darwin" ]]; then
    command_status brew
    command_status xcrun
  fi
  printf '\n'
  file_status server "$server"
  file_status grow-worker "$grow"
  file_status staging-reqs "$repo_root/apps/VC3D/mcp/requirements-staging.txt"
  file_status analysis-python "$python_env/bin/python"
  printf '\nRun vc_capabilities through an MCP client to verify runtime adapters.\n'
}

require_install_approval() {
  if [[ "${AGENTS_AGENT_MODE:-0}" == "1" && "${AGENTS_ALLOW_INSTALL:-0}" != "1" ]]; then
    fail "agent-mode install blocked; obtain approval and set AGENTS_ALLOW_INSTALL=1"
  fi
}

configure() {
  require_repo
  command -v cmake >/dev/null 2>&1 || fail "cmake is required"
  info "Configuring may fetch pinned FastMCPP dependencies from the network."
  if [[ "$(uname -s)" == "Darwin" ]]; then
    [[ -f "$build_dir/CMakeCache.txt" ]] || fail \
      "macOS cache missing; first run $repo_root/scripts/build_macos.sh (add --install-deps only with approval)"
    cmake -S "$repo_root" -B "$build_dir" \
      -DVC_BUILD_MCP_SERVER=ON \
      -DVC_TESTING=ON
  else
    (cd "$repo_root" && cmake --preset "$preset" \
      -DVC_BUILD_MCP_SERVER=ON \
      -DVC_TESTING=ON)
  fi
}

build_targets() {
  require_repo
  [[ -f "$build_dir/CMakeCache.txt" ]] || fail "configure first: $build_dir has no CMakeCache.txt"
  cmake --build "$build_dir" --target vc_mcp_server vc_grow_seg_from_seed
}

serve_http() {
  require_repo
  [[ -x "$server" ]] || fail "server not built: $server"
  [[ -x "$grow" ]] || fail "grow worker not built: $grow"
  local token="${VC_MCP_AUTH_TOKEN:-}"
  [[ ${#token} -ge 32 ]] || fail \
    "set VC_MCP_AUTH_TOKEN to at least 32 characters (review and source print-env output)"
  export VC_MCP_TRANSPORT="streamable-http"
  export VC_MCP_HOST="${VC_MCP_HOST:-127.0.0.1}"
  export VC_MCP_PORT="${VC_MCP_PORT:-18080}"
  export VC_MCP_WORK_ROOT="${VC_MCP_WORK_ROOT:-$HOME/.vc-mcp/jobs}"
  export VC_MCP_GROW_EXECUTABLE="${VC_MCP_GROW_EXECUTABLE:-$grow}"
  exec "$server"
}

test_targets() {
  require_repo
  [[ -f "$build_dir/CMakeCache.txt" ]] || fail "configure first: $build_dir has no CMakeCache.txt"
  cmake --build "$build_dir" --target \
    vc_mcp_job_store_test vc_mcp_local_worker_test vc_mcp_cpu_discovery_test \
    vc_mcp_protocol_test vc_mcp_http_transport_test
  ctest --test-dir "$build_dir" \
    -R '^(vc_mcp_job_store|vc_mcp_local_worker|vc_mcp_cpu_discovery|vc_mcp_protocol|vc_mcp_http_transport)$' \
    --output-on-failure
}

install_python() {
  require_repo
  require_install_approval
  command -v uv >/dev/null 2>&1 || fail "uv is required; installation was not attempted"
  [[ "$python_env" = /* ]] || fail "VC_MCP_PYTHON_ENV must be absolute"
  if [[ ! -x "$python_env/bin/python" ]]; then
    uv venv --python python3 "$python_env"
  fi
  uv pip install --python "$python_env/bin/python" \
    -r "$repo_root/apps/VC3D/mcp/requirements-staging.txt"
  info "Installed optional analysis/staging packages into $python_env"
  info "No model framework, repository, checkpoint, or weights were installed."
}

print_env() {
  require_repo
  cat <<EOF
# Review before sourcing. This file intentionally contains no auth token.
export VC_MCP_WORK_ROOT="${HOME}/.vc-mcp/jobs"
export VC_MCP_TIMEOUT_SECONDS="21600"
export VC_MCP_GROW_EXECUTABLE="$grow"

# Deterministic local analysis/staging adapters (optional).
export VC_MCP_ANALYSIS_PYTHON="$python_env/bin/python"
export VC_MCP_VOLUME_STAGER="$repo_root/apps/VC3D/mcp/volume_stager.py"
export VC_MCP_SURFACE_BUNDLE_ADAPTER="$repo_root/apps/VC3D/mcp/surface_bundle_adapter.py"
export VC_MCP_STRUCTURAL_EVIDENCE_ADAPTER="$repo_root/apps/VC3D/mcp/structural_evidence_adapter.py"
export VC_MCP_EVIDENCE_FUSION_ADAPTER="$repo_root/apps/VC3D/mcp/evidence_fusion_adapter.py"
export VC_MCP_REVIEW_ADAPTER="$repo_root/apps/VC3D/mcp/review_adapter.py"

# Loopback Streamable HTTP is the default for persistent agent sessions.
export VC_MCP_TRANSPORT="streamable-http"
export VC_MCP_HOST="127.0.0.1"
export VC_MCP_PORT="18080"
export VC_MCP_AUTH_TOKEN="\${VC_MCP_AUTH_TOKEN:-\$(openssl rand -hex 32)}"

# Compatibility fallback for clients without Streamable HTTP:
# export VC_MCP_TRANSPORT="stdio"
EOF
}

config_json() {
  require_repo
  python3 - <<'PY'
import json
print(json.dumps({
    "url": "http://127.0.0.1:18080/mcp",
    "headers": {
        "Authorization": "Bearer ${VC_MCP_AUTH_TOKEN}",
    },
}, indent=2))
PY
}

case "${1:-}" in
  doctor) doctor ;;
  configure) configure ;;
  build) build_targets ;;
  serve) serve_http ;;
  test) test_targets ;;
  install-python) install_python ;;
  print-env) print_env ;;
  config) config_json ;;
  -h|--help|help|'') usage ;;
  *) usage >&2; fail "unknown command: $1" ;;
esac
