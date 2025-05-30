#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

usage(){
  cat <<EOF
Usage: $0 [--start <step>]

Options:
  -s,--start <step>   One of: copy_graph, create_graph, compile, mesh, refine, surface, stitch
                      (default: copy_graph)

Steps:
  copy_graph  copy output_graph.bin back to 1352_3600_5002
  create_graph run instances_to_graph with --create_graph
  compile     compile bash script
  mesh        run graph_to_mesh
  refine      run pointcloud_mesh_refinement
  surface     run large_mesh_to_surface
  stitch      run stitch_splits finalize_mesh
EOF
  exit 1
}

# parse args
start="copy_graph"
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
  copy_graph) start_idx=1;;
  create_graph) start_idx=2;;
  compile)    start_idx=3;;
  mesh)       start_idx=4;;
  refine)     start_idx=5;;
  surface)    start_idx=6;;
  stitch)     start_idx=7;;
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

# Function to run command in Docker using the same config as start_docker.sh
run_docker_command() {
  local cmd="$1"
  
  # Use the same Docker configuration as start_docker.sh but run non-interactively
  sudo docker run --gpus all --shm-size=150g --rm \
    -v "$HOME/Desktop/scrolls:/scrolls" \
    -v "$HOME/villa/thaumato-anakalyptor/:/workspace" \
    -v "$HOME/Desktop/experiments:/workspace/experiments" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY="$DISPLAY" \
    thaumato_image \
    bash -c "cd /workspace && chmod +x compile_cpp.sh graph_labeler.sh && source /opt/conda/etc/profile.d/conda.sh && conda activate thaumato && $cmd"
}

# Function to run command in Docker with retry and restart capability
run_in_docker_with_retry() {
  local cmd="$1"
  local retry_count=0
  
  while true; do
    retry_count=$((retry_count + 1))
    echo ">>> Attempt $retry_count"
    echo ">>> Running: $cmd"
    
    # Kill any existing containers
    sudo docker kill $(sudo docker ps -q --filter ancestor=thaumato_image) 2>/dev/null || true
    sudo docker rm $(sudo docker ps -aq --filter ancestor=thaumato_image) 2>/dev/null || true
    
    # Wait a moment for cleanup
    sleep 2
    
    # Run the command using the same Docker config as start_docker.sh
    if run_docker_command "$cmd"; then
      echo ">>> SUCCESS: Command completed successfully after $retry_count attempt(s)"
      return 0
    else
      echo ">>> FAILED: Command failed (attempt $retry_count)"
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
  done
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

# step 1: copy output_graph.bin back to 1352_3600_5002
if (( start_idx <= 1 )); then
  echo ">>> Step 1: Copying output_graph.bin back to 1352_3600_5002"
  copy_graph_cmd="if [[ -f \"/workspace/experiments/${ZARR_NAME}/output_graph.bin\" ]]; then echo \"Found output_graph.bin in ${ZARR_NAME}\"; cp \"/workspace/experiments/${ZARR_NAME}/output_graph.bin\" \"/workspace/experiments/1352_3600_5002/output_graph.bin\"; elif [[ -f \"/workspace/experiments/${ZARR_NAME}_inverse/output_graph.bin\" ]]; then echo \"Found output_graph.bin in ${ZARR_NAME}_inverse\"; cp \"/workspace/experiments/${ZARR_NAME}_inverse/output_graph.bin\" \"/workspace/experiments/1352_3600_5002/output_graph.bin\"; else echo \"ERROR: output_graph.bin not found in either experiment folder\"; exit 1; fi"
  run_in_docker_with_retry "$copy_graph_cmd"
fi

# step 2: create graph
if (( start_idx <= 2 )); then
  echo ">>> Step 2: Running instances_to_graph with --create_graph"
  create_graph_cmd="python3 -m ThaumatoAnakalyptor.instances_to_graph --path \"/workspace/experiments/point_cloud_colorized_verso_subvolume_blocks\" --create_graph"
  run_in_docker_with_retry "$create_graph_cmd"
fi

# step 3: compile bash script
if (( start_idx <= 3 )); then
  echo ">>> Step 3: Compiling C++ code"
  compile_cmd="echo 'Compiling C++ code...' && ./compile_cpp.sh"
  run_in_docker_with_retry "$compile_cmd"
fi

# step 4: graph to mesh
if (( start_idx <= 4 )); then
  echo ">>> Step 4: Running graph_to_mesh"
  mesh_cmd="python3 -m ThaumatoAnakalyptor.graph_to_mesh --path /workspace/experiments/point_cloud_colorized_verso_subvolume_blocks --graph /workspace/experiments/1352_3600_5002/point_cloud_colorized_verso_subvolume_graph_BP_solved.pkl --start_point 1352 3600 5002 --angle_step 2.0 --z_spacing 5 --unfix_factor 3.0 --max_z_step_size 500 --downsample --split_width 40000"
  run_in_docker_with_retry "$mesh_cmd"
fi

# step 5: mesh refinement
if (( start_idx <= 5 )); then
  echo ">>> Step 5: Running pointcloud_mesh_refinement"
  refine_cmd="while [[ -n \"\${CONDA_DEFAULT_ENV:-}\" ]]; do conda deactivate; done && echo \"Conda completely deactivated.\" && python3 -m ThaumatoAnakalyptor.pointcloud_mesh_refinement --mesh /workspace/experiments/1352_3600_5002/mesh_0.obj --downsample_ratio 0.075"
  run_in_docker_with_retry "$refine_cmd"
fi

# step 6: large mesh to surface
if (( start_idx <= 6 )); then
  echo ">>> Step 6: Running large_mesh_to_surface"
  surface_cmd="python3 -m ThaumatoAnakalyptor.large_mesh_to_surface --input_mesh /workspace/experiments/1352_3600_5002/mesh_refined.obj --scroll /scrolls/${ZARR_NAME}.zarr --cut_size 20000 --r 16"
  run_in_docker_with_retry "$surface_cmd"
fi

# step 7: stitch splits
if (( start_idx <= 7 )); then
  echo ">>> Step 7: Running stitch_splits finalize_mesh"
  stitch_cmd="python3 -m ThaumatoAnakalyptor.stitch_splits /workspace/experiments/1352_3600_5002/working/mesh_refined.obj --type finalize_mesh --image_filename composite.jpg"
  run_in_docker_with_retry "$stitch_cmd"
fi

echo ">>> All requested steps completed for ${ZARR_NAME}."
echo ">>> Cleaning up Docker containers..."
sudo docker kill $(sudo docker ps -q --filter ancestor=thaumato_image) 2>/dev/null || true
sudo docker rm $(sudo docker ps -aq --filter ancestor=thaumato_image) 2>/dev/null || true
echo ">>> Process completed successfully!" 