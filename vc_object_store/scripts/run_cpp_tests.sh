#!/usr/bin/env bash
# Argument-free regression workflow. Persistent builds/logs stay in the output
# root; a trap removes this invocation's temporary fixtures, also on failure.
set -euo pipefail
if (( $# != 0 )); then
    echo 'This runner takes no arguments; use VC_OBJECT_STORE_TEST_ROOT.' >&2
    exit 2
fi
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
output_dir=${VC_OBJECT_STORE_TEST_ROOT:-"$source_dir/../local/vcobj-tests"}
if [[ "$output_dir" != /* || "$output_dir" == / ]]; then
    echo 'VC_OBJECT_STORE_TEST_ROOT must be an absolute non-root path.' >&2
    exit 2
fi
mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd -P)
case "$output_dir" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/private/tmp|/private/tmp/*)
        echo 'Persistent builds must not be placed in a temporary directory.' >&2
        exit 2
        ;;
esac
build_jobs=${VC_OBJECT_STORE_BUILD_JOBS:-8}
test_jobs=${VC_OBJECT_STORE_TEST_JOBS:-8}
for count in "$build_jobs" "$test_jobs"; do
    if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
        echo 'Job counts must be positive integers.' >&2
        exit 2
    fi
done
mkdir -p -- "$output_dir/build" "$output_dir/logs" "$output_dir/tmp"
fixture_dir=$(mktemp -d "$output_dir/tmp/run.XXXXXXXX")
cleanup() {
    local status=$?
    trap - EXIT
    if ! rm -rf -- "$fixture_dir"; then
        echo "Failed to clean temporary fixtures: $fixture_dir" >&2
        if (( status == 0 )); then status=1; fi
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
export TMPDIR="$fixture_dir" TMP="$fixture_dir" TEMP="$fixture_dir"
export AGENTS_AGENT_MODE=1
log="$output_dir/logs/$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
{
    printf 'Source: %s\nBuild: %s\nLog: %s\n' "$source_dir" "$output_dir/build" "$log"
    df -h "$output_dir"
    df -i "$output_dir"
    cmake -S "$source_dir" -B "$output_dir/build" -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DVC_OBJECT_STORE_BUILD_ADAPTERS=OFF \
        -DVC_OBJECT_STORE_BUILD_LINKING=OFF \
        -DVC_OBJECT_STORE_BUILD_TOOLS=OFF \
        -DVC_OBJECT_STORE_BUILD_TESTS=ON
    cmake --build "$output_dir/build" --parallel "$build_jobs"
    ctest --test-dir "$output_dir/build" --output-on-failure --parallel "$test_jobs"
} 2>&1 | tee "$log"
