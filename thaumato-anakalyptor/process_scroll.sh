#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

usage(){
  cat <<EOF
Usage: $0 [--start <step>]

Options:
  -s,--start <step>   One of: grid, instances, graph, compile, solve
                      (default: grid)

Steps:
  grid        run grid_to_pointcloud
  instances   run pointcloud_to_instances
  graph       run instances_to_graph
  compile     compile bash script
  solve       run graph_solve
  solve_inverse run graph_solve with inversed_winding_direction
  h5        run h5 instance generation
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
  compile)    start_idx=4;;
  solve)      start_idx=5;;
  solve_inverse)      start_idx=6;;
  h55)      start_idx=7;;
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
    echo "+++ Killing all python3 processes before retry…"
    pkill -9 python3 || true
    sleep 5
  done
  echo "+++ SUCCESS: $cmd_display — cleaning up any remaining python3 processes…"
  pkill -9 python3 || true
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
    --update_progress
  )
  retry_until_success "${inst_cmd[@]}"
fi

# step 3: instances_to_graph
if (( start_idx <= 3 )); then
  retry_until_success \
    python3 -m ThaumatoAnakalyptor.instances_to_graph \
    --path "/workspace/experiments/point_cloud_colorized_verso_subvolume_blocks"
fi

# step 4: compile bash script
if (( start_idx <= 4 )); then
  retry_until_success bash -c '
    echo "Compiling C++ code..."
    ./compile_cpp.sh
  '
fi

# step 5: graph_solve
if (( start_idx <= 5 )); then
  # Source the conda initialization script to define conda commands.
  source "$(conda info --base)/etc/profile.d/conda.sh"
  # Deactivate conda completely (ensure no conda environment, not even base, remains active)
  while [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; do
    conda deactivate
  done
  echo "Conda completely deactivated."
  
  python3 -m ThaumatoAnakalyptor.graph_solve \
    "/workspace/experiments/1352_3600_5002/graph.bin" \
    --experiment_name "${ZARR_NAME}"
fi

# step 6: graph_solve
if (( start_idx <= 6 )); then
  # Source the conda initialization script to define conda commands.
  source "$(conda info --base)/etc/profile.d/conda.sh"
  # Deactivate conda completely (ensure no conda environment, not even base, remains active)
  while [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; do
    conda deactivate
  done
  echo "Conda completely deactivated."
  
  python3 -m ThaumatoAnakalyptor.graph_solve \
    "/workspace/experiments/1352_3600_5002/graph.bin" \
    --experiment_name "${ZARR_NAME}_inverse" \
    --inversed_winding_direction
fi

if (( start_idx <= 7 )); then
  conda activate thaumato
  
  # python3 -m ThaumatoAnakalyptor.instances_to_h5 --input_dir /workspace/experiments/point_cloud_colorized_verso_subvolume_blocks --output_h5 /workspace/experiments/point_cloud_colorized_verso_subvolume_blocks_compact.h5 --threads 12
  retry_until_success \
    python3 -m ThaumatoAnakalyptor.instances_to_h5 \
    --input_dir "/workspace/experiments/point_cloud_colorized_verso_subvolume_blocks" \
    --output_h5 "/workspace/experiments/point_cloud_colorized_verso_subvolume_blocks_compact.h5" \
    --threads 12

fi

echo ">>> All requested steps completed for ${ZARR_NAME}."
