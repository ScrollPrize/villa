#!/usr/bin/env bash
# Runs the workflow benchmarks on origin/main's pipeline (ChunkCache) and on this
# branch's vc_reader pipeline (CPU and CUDA), after prefetching every chunk the
# scenarios read into the local cache so no run includes network time.
#
#   benchmarks/run_ab.sh [prefetch|main|cpu|cuda|compare]...   (default: all, in order)
#
# Environment: BRANCH_BUILD, MAIN_BUILD (build trees with VC_BUILD_BENCHMARKS=ON),
# VOLUME, SEGMENT, FIBERS, OUT (results folder), EXTRA (extra args for every run).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH_BUILD="${BRANCH_BUILD:-$here/../build}"
MAIN_BUILD="${MAIN_BUILD:-/mnt/raid_nvme/dev_builds/vc-main}"
DATA=/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4
VOLUME="${VOLUME:-$DATA/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr}"
# A level-2 mesh of the same volume: --segment-scale 4 maps it to level-0 voxels.
SEGMENT="${SEGMENT:-$DATA/segments/20260623142658-w028-037/mesh/20260623142658-on-20260411134726-level2.tifxyz}"
FIBERS="${FIBERS:-/mnt/raid_nvme/spiral_dataset_working/fibers}"
OUT="${OUT:-/mnt/raid_nvme/dev_builds/bench_out}"
EXTRA=(${EXTRA:-})
mkdir -p "$OUT"
JSON="$OUT/results.jsonl"

viewer_args=(--volume "$VOLUME" --segment "$SEGMENT" --segment-scale 4)
line_args=(--volume "$VOLUME" --fibers "$FIBERS")
tifxyz_args=(--volume "$VOLUME" --segment "$SEGMENT" --segment-scale 4 --group 2 --scale 1
             --slices 21 --crop 83000,22000,2048,2048 --work-dir "$OUT")

run() {  # build pipeline-args...
    local build="$1"; shift
    "$build/bin/bench_viewer_navigation" "${viewer_args[@]}" "$@" "${EXTRA[@]}"
    "$build/bin/bench_line_annotation" "${line_args[@]}" "$@" "${EXTRA[@]}"
    "$build/bin/bench_render_tifxyz" "${tifxyz_args[@]}" "$@" "${EXTRA[@]}"
}

prefetch() {
    # Replays every scenario on the s3:// source with views grown 1.5x; the second
    # round must fetch nothing (fetched_chunks=0), or the cache is still incomplete.
    for round in 1 2; do
        echo "== prefetch round $round"
        "$BRANCH_BUILD/bin/bench_viewer_navigation" "${viewer_args[@]}" --prefetch --passes 1 "${EXTRA[@]}"
        "$BRANCH_BUILD/bin/bench_line_annotation" "${line_args[@]}" --prefetch --passes 1 "${EXTRA[@]}"
    done
    "$BRANCH_BUILD/bin/bench_render_tifxyz" "${tifxyz_args[@]}" --prefetch "${EXTRA[@]}"
}

compare() {
    python3 - "$JSON" <<'EOF'
import json, sys, collections
rows = collections.OrderedDict()
pipes = []
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r["metric"] not in ("settled_ms", "first_frame_ms", "wall_ms", "ms", "presented_fps"):
        continue
    key = (r["benchmark"], r["step"], r["metric"])
    rows.setdefault(key, {})[r["pipeline"]] = r["mean"]  # the latest run of each pipeline wins
    if r["pipeline"] not in pipes:
        pipes.append(r["pipeline"])
print(f'{"benchmark":18} {"step":40} {"metric":15}' + "".join(f"{p:>16}" for p in pipes))
for (bench, step, metric), values in rows.items():
    print(f"{bench:18} {step:40} {metric:15}" + "".join(
        f'{values[p]:16.2f}' if p in values else f'{"-":>16}' for p in pipes))
EOF
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(prefetch main cpu cuda compare)
for step in "${steps[@]}"; do
    case "$step" in
        prefetch) prefetch ;;
        main) run "$MAIN_BUILD" --json "$JSON" ;;
        cpu) run "$BRANCH_BUILD" --backend cpu --json "$JSON" ;;
        cuda) run "$BRANCH_BUILD" --backend cuda --json "$JSON" ;;
        compare) compare ;;
        *) echo "unknown step $step" >&2; exit 2 ;;
    esac
done
