#!/bin/bash

# Source the conda initialization script to ensure conda commands are available.
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the 'thaumato' environment if it is not already active.
if [[ "$CONDA_DEFAULT_ENV" != "thaumato" ]]; then
    conda activate thaumato
    echo "Activated conda environment 'thaumato'."
else
    echo "Conda environment 'thaumato' is already active."
fi

source "./compile_cpp.sh"

# Deactivate conda completely (ensure no conda environment, not even 'base', remains active).
while [[ -n "$CONDA_DEFAULT_ENV" ]]; do
    conda deactivate
done
echo "Conda completely deactivated."

# Change directory to the GraphLabeler project.
cd GraphLabeler || { echo "Failed to change directory to GraphLabeler. Exiting."; exit 1; }

# Execute the main Python script.
python3 main.py