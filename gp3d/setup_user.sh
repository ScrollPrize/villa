#!/bin/bash

set -e  # Exit on error
set -x  # Print commands as they execute

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

if [ "$EUID" -eq 0 ]; then
    print_error "Please do NOT run this script as root/sudo"
    print_status "Run as a regular user to install in your home directory"
    exit 1
fi

print_status "Starting Python environment setup..."

MINICONDA_PREFIX="$HOME/miniconda3"

print_status "Installing Miniconda with Python 3.12 to $MINICONDA_PREFIX..."
wget https://repo.anaconda.com/miniconda/Miniconda3-py312_25.5.1-0-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$MINICONDA_PREFIX"
rm /tmp/miniconda.sh

# Initialize conda for current shell
eval "$($MINICONDA_PREFIX/bin/conda shell.bash hook)"

# Initialize conda for future shells
$MINICONDA_PREFIX/bin/conda init bash

print_status "Updating conda..."
conda update -n base -c defaults conda -y

print_status "Updating all conda packages..."
conda update --all -y

print_status "Updating pip and all packages..."
pip install --upgrade pip

# Update all pip packages (with error handling)
pip list --outdated --format=json | \
    python3 -c "import json, sys; print('\n'.join([x['name'] for x in json.load(sys.stdin)]))" | \
    xargs -n1 pip install -U 2>/dev/null || true

print_status "Installing PyTorch with CUDA support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


print_status "Python environment setup complete!"
print_status ""
print_status "To use the environment:"
print_status "  1. Start a new shell or run: source ~/.bashrc"
print_status "  2. Activate the environment: conda activate py312"
print_status ""
print_status "Miniconda installed at: $MINICONDA_PREFIX"