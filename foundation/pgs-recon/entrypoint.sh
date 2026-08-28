#!/bin/bash
# entrypoint.sh

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <command> [args...]"
    exit 1
fi

# Activate the virtual environment and execute the provided command
source /usr/local/educelab/pgs-recon/.venv/bin/activate
exec "$@"
