#!/bin/bash

# Function to log and exit on error
log_and_exit() {
    echo "$1"
    exit 1
}

# Ensure Docker is running
echo "Checking Docker service status..."
sudo systemctl status docker || log_and_exit "Docker service is not active. Attempting to start Docker service..."

echo "Attempting to start Docker service..."
sudo systemctl start docker || log_and_exit "Failed to start Docker service. Exiting."

echo "Checking Docker service status again..."
sudo systemctl status docker || log_and_exit "Docker service is not active after start attempt. Exiting."

echo "Docker service is running."

# Ensure Docker socket permissions
echo "Ensuring Docker socket permissions..."
sudo chown root:docker /var/run/docker.sock || log_and_exit "Failed to change Docker socket ownership. Exiting."
sudo chmod 660 /var/run/docker.sock || log_and_exit "Failed to set Docker socket permissions. Exiting."
echo "Docker socket permissions set correctly."

# Check disk space
df -h

# Ensure Conda is available
MINICONDA_PREFIX=/usr/local/miniconda
eval "$($MINICONDA_PREFIX/bin/conda shell.bash hook)"
source ~/.bashrc
export PATH=$MINICONDA_PREFIX/bin:$PATH

# Install gdown
echo "Installing gdown..."
pip install gdown || log_and_exit "Failed to install gdown. Exiting."
# Add gdown to PATH
export PATH=/home/ubuntu/.local/bin:$PATH

# Check if gdown is accessible
if ! command -v gdown &> /dev/null
then
    log_and_exit "gdown command not found. Installation might have failed."
else
    echo "gdown is installed and accessible."
fi

# Install khartes using Conda
sudo apt-get update
sudo apt-get install -y libgl1-mesa-dev || log_and_exit "Failed installing GL driver."

echo "Installing khartes using Conda..."
git clone https://github.com/KhartesViewer/khartes.git || log_and_exit "Failed to clone khartes repository. Exiting."
cd khartes || log_and_exit "Failed to change directory to khartes. Exiting."
git fetch --alls
git checkout khartes3d-beta
conda create -n khartes_env python=3.12 -y || log_and_exit "Failed to create Conda environment for khartes. Exiting."
conda activate khartes_env
pip install opencv-python || log_and_exit "Failed to install opencv. Exiting."
conda install pyqt zarr tifffile scipy pyopengl -y || log_and_exit "Failed to install libraries. Exiting."
pip install pynrrd rectpack || log_and_exit "Failed to install pynrrd and rectpack. Exiting."
conda deactivate
cd ..

# Clean up Conda caches
conda clean --all -f -y

# Check disk space
df -h

# Install volume-cartographer using Docker
echo "Installing volume-cartographer using Docker..."
sudo docker pull ghcr.io/spacegaier/volume-cartographer:edge || log_and_exit "Failed to pull Docker image for volume-cartographer. Exiting."

# Check disk space
df -h

# Install ThaumatoAnakalyptor using Docker and download checkpoints
echo "Installing ThaumatoAnakalyptor using Docker..."
git clone --recurse-submodules https://github.com/schillij95/ThaumatoAnakalyptor.git || log_and_exit "Failed to clone ThaumatoAnakalyptor repository. Exiting."
cd ThaumatoAnakalyptor || log_and_exit "Failed to change directory to ThaumatoAnakalyptor. Exiting."

# Download checkpoints
# Create directories if they don't exist
mkdir -p ./ThaumatoAnakalyptor/mask3d/saved/train/
mkdir -p ./Vesuvius-Grandprize-Winner/
gdown --id 1gO8Nf4sCaA7r4dO6ePtt0SE0E5ePXSid -O ./ThaumatoAnakalyptor/mask3d/saved/train/last-epoch.ckpt || log_and_exit "Failed to download checkpoint 1 for ThaumatoAnakalyptor. Exiting."
gdown --id 13Iu-dR-1sKq_oGJfNa86LcBSv1o4XA37 -O ./Vesuvius-Grandprize-Winner/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt || log_and_exit "Failed to download checkpoint 2 for ThaumatoAnakalyptor. Exiting."

# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
&& curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Update package lists
sudo apt-get update

# Install NVIDIA Docker packages
sudo apt-get install -y nvidia-docker2 nvidia-container-runtime || log_and_exit "Failed to install NVIDIA Docker dependencies. Exiting."
sudo systemctl restart docker || log_and_exit "Failed to restart Docker service after installing NVIDIA dependencies. Exiting."

# Checking space
df -h

# Uncomment to install Thaumato on GPU Instance (can work ok CPU?)
sudo docker pull scrollprize/thaumato_image:latest|| log_and_exit "Failed to pull Docker image for tomato. Exiting."
cd ..

# Check disk space
df -h

# Install Crackle-Viewer
echo "Installing Crackle-Viewer..."
git clone https://github.com/schillij95/Crackle-Viewer.git || log_and_exit "Failed to clone Crackle-Viewer repository. Exiting."
cd Crackle-Viewer || log_and_exit "Failed to change directory to Crackle-Viewer. Exiting."
conda create -n crackle_env python=3.8 -y || log_and_exit "Failed to create Conda environment for Crackle-Viewer. Exiting."
conda activate crackle_env || log_and_exit "Failed to activate Conda environment for Crackle-Viewer. Exiting."
conda install tk -c conda-forge -y || log_and_exit "Failed to install tk. Exiting."
pip install -r requirements.txt || log_and_exit "Failed to install requirements for Crackle-Viewer. Exiting."
conda deactivate
cd ..

# Clean up Conda caches
conda clean --all -f -y

# Check disk space
df -h

# Install Meshlab
echo "Installing Meshlab..."
sudo apt-get update || log_and_exit "Failed to update package list for Meshlab. Exiting."
sudo apt-get install -y meshlab || log_and_exit "Failed to install Meshlab. Exiting."

# Vesuvius
echo "Installing vesuvius scaffold..."
conda create -n vesuvius python=3.12 -y || log_and_exit "Failed to create Vesuvius environment"
conda activate vesuvius
conda install jupyter matplotlib
pip install vesuvius

echo "Installation completed!"
