#!/bin/bash

# Update the system
sudo apt-get update
sudo apt-get upgrade -y --with-new-pkgs

# Install prerequisites
sudo apt-get install -y wget bzip2 curl git

# Check disk space
df -h

# Install Miniconda
MINICONDA_INSTALLER_SCRIPT=Miniconda3-latest-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local/miniconda
wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT
sudo ./$MINICONDA_INSTALLER_SCRIPT -b -p $MINICONDA_PREFIX
rm $MINICONDA_INSTALLER_SCRIPT

# Initialize conda for the current session and add it to PATH
eval "$($MINICONDA_PREFIX/bin/conda shell.bash hook)"
conda init
source ~/.bashrc

# Export PATH for future steps
export PATH=$MINICONDA_PREFIX/bin:$PATH

# Check disk space
df -h

# Install Docker
sudo apt-get remove -y docker docker-engine docker.io containerd runc
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add the default user to the docker group
sudo groupadd docker
sudo usermod -aG docker $USER

# Apply group changes
newgrp docker <<EOF
  # Change permissions of Docker socket
  sudo chown root:docker /var/run/docker.sock
  sudo chmod 660 /var/run/docker.sock

  # Restart Docker service
  sudo systemctl restart docker

  # Confirm Docker is running and available
  docker --version
EOF


echo "Docker and Conda installation completed. Docker and Conda are now available in the PATH."
