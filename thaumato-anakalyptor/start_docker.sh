#!/bin/bash
# Filename: run_thaumato.sh

# Change directory to the repository and update it
cd "$HOME/villa" || { echo "Directory $HOME/villa not found!"; exit 1; }
git pull

# Enable X11 access for Docker containers and root
xhost +local:docker
xhost +local:root

# Run the Docker container with GPU access, large shared memory, multiple volume mounts, and GUI support.
sudo docker run --gpus all --shm-size=150g -it --rm \
    -v "$HOME/Desktop/scrolls:/scrolls" \
    -v "$HOME/villa/thaumato-anakalyptor/:/workspace" \
    -v "$HOME/Desktop/experiments:/workspace/experiments" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY="$DISPLAY" \
    thaumato_image \
    bash -c "chmod +x compile_cpp.sh graph_labeler.sh ; exec bash"
