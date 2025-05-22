#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

# find zarr…
ZARR_DIR=$(find /scrolls -maxdepth 1 -type d -name '*.zarr' | head -n1)
[[ -n "$ZARR_DIR" ]] || { echo "No .zarr found under /scrolls" >&2; exit 1; }
ZARR_NAME=$(basename "$ZARR_DIR" .zarr)

retry_until_success() {
  local args=( "$@" )
  local cmd_display; printf -v cmd_display '%q ' "${args[@]}"
  until "${args[@]}"; do
    echo "+++ FAILED: $cmd_display — retrying in 5s…"
    sleep 5
  done
}

# step 1
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

# step 2
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

# step 3
retry_until_success \
  python3 -m ThaumatoAnakalyptor.instances_to_graph \
  --path "/workspace/experiments/point_cloud_colorized_verso_subvolume_blocks"

# deactivate
conda deactivate || echo "(already deactivated?)"
conda deactivate || echo "(still out?)"

# final solve
solve_cmd=(
  python3 -m ThaumatoAnakalyptor.graph_solve
  "/workspace/experiments/1352_3600_5002/graph.bin"
  --experiment_name "${ZARR_NAME}"
)
retry_until_success "${solve_cmd[@]}"

echo "All done for $ZARR_NAME!"
