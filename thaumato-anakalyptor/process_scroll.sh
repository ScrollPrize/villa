#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

# 1) find the .zarr
ZARR_DIR=$(find /scroll -maxdepth 1 -type d -name '*.zarr' | head -n1)
if [[ -z "$ZARR_DIR" ]]; then
  echo "ERROR: no .zarr directory found under /scroll" >&2
  exit 1
fi
ZARR_NAME=$(basename "$ZARR_DIR" .zarr)
echo ">>> Found Zarr: $ZARR_NAME (dir: $ZARR_DIR)"

# helper to retry until success
retry_until_success() {
  local cmd="$*"
  until eval "$cmd"; do
    echo "+++ Command failed, retrying in 5s: $cmd"
    sleep 5
  done
}

# 2a) grid_to_pointcloud
retry_until_success python3 -m ThaumatoAnakalyptor.grid_to_pointcloud \
    --base_path "" \
    --volume_subpath "/scroll/${ZARR_NAME}.zarr" \
    --disk_load_save "" "" \
    --pointcloud_subpath "/scroll_pcs/point_cloud" \
    --num_threads 20 \
    --gpus 4

# 2b) pointcloud_to_instances
retry_until_success python3 -m ThaumatoAnakalyptor.pointcloud_to_instances \
    --path "/scroll_pcs" \
    --dest "/scroll_pcs" \
    --umbilicus_path "/scroll_pcs/umbilicus.txt" \
    --main_drive "" \
    --alternative_ply_drives "" "" \
    --batch_size 12 \
    --gpus 4

# 2c) instances_to_graph
retry_until_success python3 -m ThaumatoAnakalyptor.instances_to_graph \
    --path "/scroll_pcs/point_cloud_colorized_verso_subvolume_blocks"

# 3) deactivate conda twice (ignore failures)
conda deactivate || echo "(first conda deactivate failed, continuing)"
conda deactivate || echo "(second conda deactivate failed, continuing)"

# 4) graph_solve
retry_until_success python3 -m ThaumatoAnakalyptor.graph_solve \
    "/scroll_pcs/1352_3600_5002/graph.bin" \
    --experiment_name "${ZARR_NAME}"

echo ">>> All steps completed successfully for ${ZARR_NAME}."
