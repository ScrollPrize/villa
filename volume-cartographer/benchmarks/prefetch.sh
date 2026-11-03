#!/usr/bin/env bash
# Fills the benchmark cache (benchmarks.json "cache"; default: cache/ beside this
# script) with everything the cases read, so cache state "cached" runs need no
# network: remote segments and fiber folders are downloaded, then every case's
# scenarios are replayed on the volume's remote source with views grown by
# prefetch_margin (1.5), filling the cached copy of the volume. A second replay
# round warns about anything it still had to fetch. Needs a vc_reader build.
#
#   benchmarks/prefetch.sh [viewer_navigation|line_annotation|render_tifxyz]...   (default: all)
#
# Environment: CONFIG (the manifest), BUILD (build tree, default ../build),
# EXTRA (extra arguments for every run).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$(realpath "${CONFIG:-$here/benchmarks.json}")"
BUILD="${BUILD:-$here/../build}"
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
        for round in 1 2; do
            echo "== prefetch $bench / $case, round $round"
            check=(); [ $round -eq 2 ] && check=(--expect-cached)
            "$bin" --config "$CONFIG" --case "$case" --prefetch --passes 1 "${check[@]}" "${EXTRA[@]}"
        done
    done
done
python3 -c 'import json, os, sys
c = sys.argv[1]
print("cache:", os.path.normpath(os.path.join(os.path.dirname(c), json.load(open(c)).get("cache", "cache"))))' "$CONFIG"
