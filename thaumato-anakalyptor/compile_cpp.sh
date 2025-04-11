#!/bin/bash

# Source the conda initialization script to define conda commands.
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the 'thaumato' environment if it is not already active.
if [[ "$CONDA_DEFAULT_ENV" != "thaumato" ]]; then
    conda activate thaumato
    echo "Activated conda environment 'thaumato'."
else
    echo "Conda environment 'thaumato' is already active."
fi

# Build the sheet_generation project.
cd ThaumatoAnakalyptor/sheet_generation || exit 1
mkdir -p build
cd build || exit 1
cmake ..
cmake --build . --parallel 32

# Build the graph_problem project.
cd /workspace/ThaumatoAnakalyptor/graph_problem || exit 1
mkdir -p build
cd build || exit 1

# Deactivate conda completely (ensure no conda environment, not even 'base', remains active).
while [[ -n "$CONDA_DEFAULT_ENV" ]]; do
    conda deactivate
done
echo "Conda completely deactivated."

cmake ..
cmake --build . --parallel 32
