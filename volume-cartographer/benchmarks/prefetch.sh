#!/usr/bin/env bash
# Fills the benchmark cache (benchmarks.json "cache"; default: cache/ beside this
# script) with everything the cases read, so cache state "cached" runs need no
# network. Needs a vc_reader build. Already cached data is never fetched again.
#
#   benchmarks/prefetch.sh [viewer_navigation|line_annotation|render_tifxyz]...   (default: all)
#
# - Remote segments and fiber folders are downloaded (files already there are kept).
# - Viewer harnesses plan instead of replaying online: each pass drives the case's
#   inputs on the cache offline (remote_offline "fill": a chunk the cache lacks reads
#   as empty at once and is recorded), then fetches exactly the recorded chunks.
#   Passes repeat until one finds nothing missing (exit 0), which is also all a pass
#   costs when the cache is already complete. Each case is planned exactly as its
#   timed runs read (margin 1, its own passes) on every backend in BACKENDS (CUDA
#   sampling reads chunks the CPU does not), then with views grown by
#   prefetch_margin (1.5) to cover other pipelines' level and fallback choices. Grown
#   views pan to other places, so they do not stand in for the exact plans.
# - render_tifxyz runs the tool once online through the cache.
#
# Environment: CONFIG (the manifest), BUILD (build tree, default ../build),
# BACKENDS (exact plans, default "cpu cuda"), MAX_PASSES (plan passes per plan,
# default 6), EXTRA (extra arguments for every run).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$(realpath "${CONFIG:-$here/benchmarks.json}")"
BUILD="${BUILD:-$here/../build}"
MAX_PASSES="${MAX_PASSES:-6}"
BACKENDS=(${BACKENDS:-cpu cuda})
EXTRA=(${EXTRA:-})
benches=("$@")
[ ${#benches[@]} -eq 0 ] && benches=(viewer_navigation line_annotation render_tifxyz)

for bench in "${benches[@]}"; do
    bin="$BUILD/bin/bench_$bench"
    for case in $("$bin" --config "$CONFIG" --list-cases); do
        if [ "$bench" = render_tifxyz ]; then
            echo "== prefetch $bench / $case"
            "$bin" --config "$CONFIG" --case "$case" --prefetch "${EXTRA[@]}"
            continue
        fi
        plans=()
        for backend in "${BACKENDS[@]}"; do plans+=("exact-$backend"); done
        plans+=(margin)
        for plan in "${plans[@]}"; do
            if [ $plan = margin ]; then
                args=(--passes 1)                             # the manifest's prefetch_margin
            else
                args=(--prefetch-margin 1 --backend "${plan#exact-}")  # what timed runs read
            fi
            for ((pass = 1; ; ++pass)); do
                echo "== prefetch $bench / $case, $plan plan pass $pass"
                status=0
                "$bin" --config "$CONFIG" --case "$case" --prefetch "${args[@]}" "${EXTRA[@]}" || status=$?
                [ $status -eq 0 ] && break                    # nothing missing: complete
                if [ $status -ne 2 ]; then                    # 2: fetched what it lacked
                    echo "prefetch: $bench / $case failed (exit $status)" >&2
                    exit $status
                fi
                if [ $pass -ge "$MAX_PASSES" ]; then
                    echo "prefetch: $bench / $case still lacked chunks after $pass passes" >&2
                    exit 1
                fi
            done
        done
    done
done
python3 -c 'import json, os, sys
c = sys.argv[1]
print("cache:", os.path.normpath(os.path.join(os.path.dirname(c), json.load(open(c)).get("cache", "cache"))))' "$CONFIG"
