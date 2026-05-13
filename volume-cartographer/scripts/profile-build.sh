#!/usr/bin/env bash
# Compile-time profiling helper. Builds with the dev-profile-clang preset
# (clang -ftime-trace) and prints the slowest TUs from ninja's .ninja_log.
#
# Usage:
#   scripts/profile-build.sh                        # full clean build + summary
#   scripts/profile-build.sh report <build-dir>     # just the report from an existing log
#
# After running, the slowest .o files have a sibling <name>.json. Open in
# chrome://tracing or speedscope:
#   open https://www.speedscope.app  → load the .json
# For a whole-build aggregate ("worst includes / templates / functions"),
# install ClangBuildAnalyzer and run:
#   ClangBuildAnalyzer --all build/dev-profile-clang capture.bin
#   ClangBuildAnalyzer --analyze capture.bin

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Print the top-N slowest objects from a .ninja_log file. Format is:
#   start_ms  end_ms  restat_mtime  output  command_hash
ninja_log_top() {
    local log=$1 top=${2:-30}
    awk 'NR>1 && NF>=4 { dur = $2 - $1; if (dur > 0) printf "%6d ms  %s\n", dur, $4 }' "$log" \
        | sort -rn | head -"$top"
}

cmd_build() {
    local build_dir=build/dev-profile-clang
    rm -rf "$build_dir"
    cmake --preset dev-profile-clang
    cmake --build --preset dev-profile-clang
    cmd_report "$build_dir"
}

cmd_report() {
    local build_dir=${1:-build/dev-profile-clang}
    local log="$build_dir/.ninja_log"
    if [[ ! -f "$log" ]]; then
        echo "no $log — run a build first" >&2
        return 1
    fi
    echo "=== Top 30 slowest build steps (ms) — $log ==="
    ninja_log_top "$log" 30
    echo
    local n_traces
    n_traces=$(find "$build_dir" -name '*.cpp.o.json' 2>/dev/null | wc -l)
    if (( n_traces > 0 )); then
        echo "=== Per-TU compile traces ==="
        echo "Found $n_traces .json trace files under $build_dir"
        echo "Open one in chrome://tracing or https://www.speedscope.app"
        echo
        echo "ClangBuildAnalyzer (optional, https://github.com/aras-p/ClangBuildAnalyzer):"
        echo "  ClangBuildAnalyzer --all $build_dir capture.bin && ClangBuildAnalyzer --analyze capture.bin"
    else
        echo "(no -ftime-trace .json files found — was the build configured with -DVC_PROFILE_COMPILE=ON or via the dev-profile-clang preset?)"
    fi
}

case "${1:-build}" in
    build)  cmd_build ;;
    report) shift; cmd_report "$@" ;;
    *)
        echo "Usage: $0 [build | report <build-dir>]" >&2
        exit 1
        ;;
esac
