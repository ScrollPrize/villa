#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

usage(){
  cat <<EOF
Usage: $0 [--start <step>]

Options:
  -s,--start <step>   One of: grid, instances, graph, compile, solve, solve_inverse, h5
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
  h5)      start_idx=7;;
  *) echo "Invalid start step: $start"; usage;;
esac

# Docker setup function
setup_docker() {
  echo ">>> Setting up Docker environment..."
  cd "$HOME/villa" || { echo "Directory $HOME/villa not found!"; exit 1; }
  git pull
  
  # Enable X11 access for Docker containers and root
  xhost +local:docker
  xhost +local:root
}

# Function to run command in Docker with retry and restart capability
run_in_docker_with_retry() {
  local cmd="$1"
  local max_retries=3
  local retry_count=0
  
  while (( retry_count < max_retries )); do
    echo ">>> Attempt $((retry_count + 1)) of $max_retries"
    echo ">>> Running: $cmd"
    
    # Kill any existing containers
    sudo docker kill $(sudo docker ps -q --filter ancestor=thaumato_image) 2>/dev/null || true
    sudo docker rm $(sudo docker ps -aq --filter ancestor=thaumato_image) 2>/dev/null || true
    
    # Wait a moment for cleanup
    sleep 2
    
    # Run the command in Docker
    if sudo docker run --gpus all --shm-size=150g --rm \
        -v "$HOME/Desktop/scrolls:/scrolls" \
        -v "$HOME/villa/thaumato-anakalyptor/:/workspace" \
        -v "$HOME/Desktop/experiments:/workspace/experiments" \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY="$DISPLAY" \
        thaumato_image \
        bash -c "cd /workspace && $cmd"; then
      echo ">>> SUCCESS: Command completed successfully"
      return 0
    else
      echo ">>> FAILED: Command failed (attempt $((retry_count + 1)))"
      retry_count=$((retry_count + 1))
      
      if (( retry_count < max_retries )); then
        echo ">>> Cleaning up and restarting Docker in 5 seconds..."
        
        # Kill all Docker containers
        sudo docker kill $(sudo docker ps -q) 2>/dev/null || true
        sudo docker rm $(sudo docker ps -aq) 2>/dev/null || true
        
        # Restart Docker service
        sudo systemctl restart docker
        sleep 10
        
        # Re-setup Docker environment
        setup_docker
        
        echo ">>> Docker restarted, retrying command..."
      fi
    fi
  done
  
  echo ">>> ERROR: Command failed after $max_retries attempts"
  exit 1
}

# Find the .zarr directory from host
echo ">>> Looking for .zarr directory..."
ZARR_DIR=$(find "$HOME/Desktop/scrolls" -maxdepth 1 -type d -name '*.zarr' | head -n1)
if [[ -z "$ZARR_DIR" ]]; then
  echo "ERROR: no .zarr directory found under $HOME/Desktop/scrolls" >&2
  exit 1
fi
ZARR_NAME=$(basename "$ZARR_DIR" .zarr)
echo ">>> Processing Zarr: $ZARR_NAME"

# Initial Docker setup
setup_docker

# step 1: grid_to_pointcloud
if (( start_idx <= 1 )); then
  echo ">>> Step 1: Running grid_to_pointcloud"
  grid_cmd="python3 -m ThaumatoAnakalyptor.grid_to_pointcloud --base_path \"\" --volume_subpath \"/scrolls/${ZARR_NAME}.zarr\" --disk_load_save \"\" \"\" --pointcloud_subpath \"/workspace/experiments/point_cloud\" --num_threads 20 --gpus 4"
  run_in_docker_with_retry "$grid_cmd"
fi

# step 2: pointcloud_to_instances
if (( start_idx <= 2 )); then
  echo ">>> Step 2: Running pointcloud_to_instances"
  inst_cmd="python3 -m ThaumatoAnakalyptor.pointcloud_to_instances --path \"/workspace/experiments\" --dest \"/workspace/experiments\" --umbilicus_path \"/workspace/experiments/umbilicus.txt\" --main_drive \"\" --alternative_ply_drives \"\" \"\" --batch_size 5 --gpus 4 --update_progress"
  run_in_docker_with_retry "$inst_cmd"
fi

# step 3: instances_to_graph
if (( start_idx <= 3 )); then
  echo ">>> Step 3: Running instances_to_graph"
  graph_cmd="python3 -m ThaumatoAnakalyptor.instances_to_graph --path \"/workspace/experiments/point_cloud_colorized_verso_subvolume_blocks\""
  run_in_docker_with_retry "$graph_cmd"
fi

# step 4: compile bash script
if (( start_idx <= 4 )); then
  echo ">>> Step 4: Compiling C++ code"
  compile_cmd="chmod +x compile_cpp.sh && ./compile_cpp.sh"
  run_in_docker_with_retry "$compile_cmd"
fi

# step 5: graph_solve
if (( start_idx <= 5 )); then
  echo ">>> Step 5: Running graph_solve"
  solve_cmd="source \"\$(conda info --base)/etc/profile.d/conda.sh\" && while [[ -n \"\${CONDA_DEFAULT_ENV:-}\" ]]; do conda deactivate; done && echo \"Conda completely deactivated.\" && python3 -m ThaumatoAnakalyptor.graph_solve \"/workspace/experiments/1352_3600_5002/graph.bin\" --experiment_name \"${ZARR_NAME}\""
  run_in_docker_with_retry "$solve_cmd"
fi

# step 6: graph_solve with inverse
if (( start_idx <= 6 )); then
  echo ">>> Step 6: Running graph_solve with inverse winding direction"
  solve_inverse_cmd="source \"\$(conda info --base)/etc/profile.d/conda.sh\" && while [[ -n \"\${CONDA_DEFAULT_ENV:-}\" ]]; do conda deactivate; done && echo \"Conda completely deactivated.\" && python3 -m ThaumatoAnakalyptor.graph_solve \"/workspace/experiments/1352_3600_5002/graph.bin\" --experiment_name \"${ZARR_NAME}_inverse\" --inversed_winding_direction"
  run_in_docker_with_retry "$solve_inverse_cmd"
fi

# step 7: h5 generation
if (( start_idx <= 7 )); then
  echo ">>> Step 7: Running h5 instance generation"
  h5_cmd="python3 -m ThaumatoAnakalyptor.instances_to_h5 --input_dir \"/workspace/experiments/point_cloud_colorized_verso_subvolume_blocks\" --output_h5 \"/workspace/experiments/point_cloud_colorized_verso_subvolume_blocks_compact.h5\" --threads 12"
  run_in_docker_with_retry "$h5_cmd"
fi

echo ">>> All requested steps completed for ${ZARR_NAME}."
echo ">>> Cleaning up Docker containers..."
sudo docker kill $(sudo docker ps -q --filter ancestor=thaumato_image) 2>/dev/null || true
sudo docker rm $(sudo docker ps -aq --filter ancestor=thaumato_image) 2>/dev/null || true
echo ">>> Process completed successfully!" 