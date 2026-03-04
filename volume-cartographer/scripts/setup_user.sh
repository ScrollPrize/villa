#!/bin/bash
set -e

# Determine architecture and download appropriate Miniconda installer
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py312_25.5.1-1-Linux-x86_64.sh"
elif [ "$ARCH" = "aarch64" ]; then
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py312_25.5.1-1-Linux-aarch64.sh"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# Download and install Miniconda
MINICONDA_INSTALLER="Miniconda3-installer.sh"
echo "Downloading Miniconda for $ARCH..."
wget -q "$MINICONDA_URL" -O "$MINICONDA_INSTALLER"

echo "Installing Miniconda..."
bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
rm "$MINICONDA_INSTALLER"

# Initialize conda for bash
"$HOME/miniconda3/bin/conda" init bash

# Add environment variables to .bashrc
echo "" >> ~/.bashrc
echo "# Vesuvius Challenge environment variables" >> ~/.bashrc
echo "export WANDB_ENTITY=\"vesuvius-challenge\"" >> ~/.bashrc

# Source .bashrc to activate changes
source ~/.bashrc

echo "Miniconda installation complete!"
echo "Environment variables added to ~/.bashrc"
echo "Please restart your terminal or run 'source ~/.bashrc' to activate conda"