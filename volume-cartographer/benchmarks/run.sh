#!/usr/bin/env bash
# Runs the workflow benchmarks: every case of benchmarks.json (or $CONFIG) for each
# harness, on this branch's vc_reader pipeline (CPU and CUDA) and, when asked, on
# origin/main's pipeline (ChunkCache) for comparison. "compare" lays the results out
# with the manifest's saved "results" and their "baseline" (compare.py), so a
# baseline need not be rerun. Run prefetch.sh first.
#
#   benchmarks/run.sh [cpu|cuda|main|compare]...   (default: cpu cuda compare)
#
# Environment:
#   CONFIG        the manifest (default: benchmarks.json beside this script)
#   STATES        cache states to run (default "cached"; any of "cached partial cold")
#   BENCHES       harnesses (default "viewer_navigation line_annotation render_tifxyz")
#   BRANCH_BUILD  this branch's build tree (default ../build), VC_BUILD_BENCHMARKS=ON
#   MAIN_BUILD    origin/main's build tree for the "main" step (see README.md)
#   OUT           results folder (default: out/ beside this script)
#   EXTRA         extra arguments for every run, e.g. "--passes 1 --pan-steps 2"
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$(realpath "${CONFIG:-$here/benchmarks.json}")"
STATES=(${STATES:-cached})
BENCHES=(${BENCHES:-viewer_navigation line_annotation render_tifxyz})
BRANCH_BUILD="${BRANCH_BUILD:-$here/../build}"
MAIN_BUILD="${MAIN_BUILD:-/mnt/raid_nvme/dev_builds/vc-main}"
OUT="${OUT:-$here/out}"
EXTRA=(${EXTRA:-})
mkdir -p "$OUT"
JSON="$OUT/results.jsonl"

run() {  # label build pipeline-args...
    local label="$1" build="$2"; shift 2
    for state in "${STATES[@]}"; do
        for bench in "${BENCHES[@]}"; do
            local bin="$build/bin/bench_$bench" extra=()
            [ "$bench" = render_tifxyz ] && extra=(--work-dir "$OUT/work")
            for case in $("$bin" --config "$CONFIG" --list-cases); do
                echo "== $bench / $case [$label, $state]"
                "$bin" --config "$CONFIG" --case "$case" --cache-state "$state" --json "$JSON" \
                       "${extra[@]}" "$@" "${EXTRA[@]}"
            done
        done
    done
}

compare() {
    python3 "$here/compare.py" --config "$CONFIG" "$JSON"
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(cpu cuda compare)
for step in "${steps[@]}"; do
    case "$step" in
        cpu) run vc_reader-cpu "$BRANCH_BUILD" --backend cpu ;;
        cuda) run vc_reader-cuda "$BRANCH_BUILD" --backend cuda ;;
        main)
            # origin/main's tree only reads the cached copies (no remote reads).
            saved=("${STATES[@]}"); STATES=(cached)
            [ "${saved[*]}" != cached ] && echo "main: running cache state 'cached' only" >&2
            run chunkcache "$MAIN_BUILD"
            STATES=("${saved[@]}") ;;
        compare) compare ;;
        *) echo "unknown step $step" >&2; exit 2 ;;
    esac
done
