#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

usage(){
  cat <<EOF
Usage: $0 [--start <step>]

Options:
  -s,--start <step>   One of: grid, instances, graph, conda, solve
                      (default: grid)

Steps:
  grid        run grid_to_pointcloud
  instances   run pointcloud_to_instances
  graph       run instances_to_graph
  conda       run the two conda deactivate calls
  solve       run graph_solve
EOF
  exit 1
}

# parse args
start="grid"
while [[ $# -gt 0 ]]; do
  case $1 in
    -s|--start)
      start="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Unknown option: $1"; usage;;
  esac
done

# map step name to index
case "$start" in
  grid)       start_idx=1;;
  instances)  start_idx=2;;
  graph)      start_idx=3;;
  conda)      start_idx=4;;
  solve)      start_idx=5;;
  *) echo "Invalid start step: $start"; usage;;
esac

# find the .zarr directory
ZARR_DIR=$(find /scrolls -maxdepth 1 -type d -name '*.zarr' | head -n1)
if [[ -z "$ZARR_DIR" ]]; then
  echo "ERROR: no .zarr directory found under /scrolls" >&2
  exit 1
fi
ZARR_NAME=$(basename "$ZARR_DIR" .zarr)
echo ">>> Processing Zarr: $ZARR_NAME"

# retry helper (preserves empty-string args & logs them verbatim)
retry_until_success(){
  local args=( "$@" )
  local cmd_display
  printf -v cmd_display '%q ' "${args[@]}"
  until "${args[@]}"; do
    echo "+++ FAILED: $cmd_display — retrying in 5s…"
    sleep 5
  done
}

# step 1: grid_to_pointcloud
if (( start_idx <= 1 )); then
  grid_cmd=(
    python3 -m ThaumatoAnakalyptor.grid_to_pointcloud
    --base_path ""
    --volume_subpath "/scrolls/${ZARR_NAME}.zarr"
    --disk_load_save "" ""
    --pointcloud_subpath "/workspace/experiments/point_cloud"
    --num_threads 20
    --gpus 4
  )
  retry_until_success "${grid_cmd[@]}"
fi

# step 2: pointcloud_to_instances
if (( start_idx <= 2 )); then
  inst_cmd=(
    python3 -m ThaumatoAnakalyptor.pointcloud_to_instances
    --path "/workspace/experiments"
    --dest "/workspace/experiments"
    --umbilicus_path "/workspace/experiments/umbilicus.txt"
    --main_drive ""
    --alternative_ply_drives "" ""
    --batch_size 5
    --gpus 4
  )
  retry_until_success "${inst_cmd[@]}"
fi

# step 3: instances_to_graph
if (( start_idx <= 3 )); then
  retry_until_success \
    python3 -m ThaumatoAnakalyptor.instances_to_graph \
    --path "/workspace/experiments/point_cloud_colorized_verso_subvolume_blocks"
fi

# step 4: conda deactivate twice
if (( start_idx <= 4 )); then
  conda deactivate || echo "(first conda deactivate failed, continuing)"
  conda deactivate || echo "(second conda deactivate failed, continuing)"
fi

# step 5: graph_solve
if (( start_idx <= 5 )); then
  solve_cmd=(
    python3 -m ThaumatoAnakalyptor.graph_solve
    "/workspace/experiments/1352_3600_5002/graph.bin"
    --experiment_name "${ZARR_NAME}"
  )
  retry_until_success "${solve_cmd[@]}"
fi

echo ">>> All requested steps completed for ${ZARR_NAME}."
