#!/bin/bash
# Filename: run_thaumato.sh

# Change directory to the repository and update it
cd /home/ubuntu/villa || { echo "Directory /home/ubuntu/villa not found!"; exit 1; }
git pull

# Enable X11 access for Docker containers and root
xhost +local:docker
xhost +local:root

# Run the Docker container with GPU access, large shared memory, multiple volume mounts, and GUI support.
sudo docker run --gpus all --shm-size=150g -it --rm \
    -v /home/ubuntu/Desktop/scrolls:/scrolls \
    -v /home/ubuntu/villa/thaumato-anakalyptor/:/workspace \
    -v /home/ubuntu/Desktop/experiments:/workspace/experminets \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    thaumato_image \
    bash -c "chmod +x compile_cpp.sh && ./compile_cpp.sh && chmod +x GraphLabeler.sh && ./GraphLabeler.sh"
