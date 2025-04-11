#!/bin/bash
# Filename: run_thaumato.sh

# Enable X11 access for the docker user and root
xhost +local:docker
xhost +local:root

# Run the Docker container with GPU support and mounts,
# then execute the commands to change script permissions and run them.
docker run --gpus all --shm-size=150g -it --rm \
    -v /home/ubuntu/Desktop/scrolls:/scrolls \
    -v /home/ubuntu/villa/thaumato-anakalyptor/:/workspace \
    -v /home/ubuntu/Desktop/experiments:/workspace/experminets \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    thaumato_image \
    bash -c "chmod +x /workspace/compile_cpp.sh && /workspace/compile_cpp.sh && chmod +x /workspace/GraphLabeler.sh && /workspace/GraphLabeler.sh"