#!/bin/bash
set -euo pipefail

echo "============================================================"
echo " STEP 1: DOWNLOADING REAL SCROLL DATA (PHerc0009B Subset)"
echo "============================================================"
# The user must provide their dl.ash2txt.org credentials via HTTP basic auth if required.
DATA_DIR="pherc0009b_evidence_data"
mkdir -p "$DATA_DIR"

# Using a standard path subset structure for Scroll3 (PHerc0009B)
URL_PREFIX="http://dl.ash2txt.org/full-scrolls/Scroll3.volpkg/paths"
WGET_OPTS="--no-parent -r -nH --cut-dirs=3 --reject='index.html*' -P ${DATA_DIR}"

if [[ -n "${DL_USER:-}" && -n "${DL_PASS:-}" ]]; then
    WGET_OPTS="--user=${DL_USER} --password=${DL_PASS} ${WGET_OPTS}"
fi

# Download a specific track
TARGET_TRACK="${PHERC0009B_TRACK_ID:-20231031143822}" # A known track ID for Scroll3

echo "Attempting to download PHerc0009B sample data from $URL_PREFIX/${TARGET_TRACK}..."

# We support a zipped community bundle if direct track download isn't available
ZIP_URL="https://dl.ash2txt.org/community-uploads/paul/pherc0009b_evidence_subset.zip"

if wget -q --spider "$ZIP_URL"; then
    echo "Downloading pre-packaged evidence subset..."
    wget -c "$ZIP_URL" -O "${DATA_DIR}/pherc0009b_evidence_subset.zip"
    unzip -q -o "${DATA_DIR}/pherc0009b_evidence_subset.zip" -d "${DATA_DIR}"
    TRACK_DIR="${DATA_DIR}/pherc0009b_evidence_subset"
else
    echo "Pre-packaged subset not found. Attempting direct path download..."
    # Suppress output to avoid spamming the console, only show errors
    wget $WGET_OPTS "${URL_PREFIX}/${TARGET_TRACK}/" || {
        echo "WARNING: Download failed. You may need to export DL_USER and DL_PASS, or specify a valid PHERC0009B_TRACK_ID."
        echo "Continuing in case data is already mounted or cached locally."
    }
    TRACK_DIR="${DATA_DIR}/paths/${TARGET_TRACK}"
fi

if [[ ! -d "$TRACK_DIR" ]]; then
    echo "CRITICAL ERROR: No track data found at $TRACK_DIR."
    echo "Please download the data manually or provide valid credentials."
    exit 1
fi

echo "Data located at: $TRACK_DIR"

echo "============================================================"
echo " STEP 2: EXECUTING EVIDENCE PIPELINE (run_real_scroll_evidence.py)"
echo "============================================================"

# Ensure the python script is executable and available
PYTHON_SCRIPT="run_real_scroll_evidence.py"
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "CRITICAL ERROR: $PYTHON_SCRIPT not found in current directory."
    exit 1
fi

# Run the python pipeline which handles Baseline -> Upgraded -> spiralcheck (C4)
python "$PYTHON_SCRIPT" \
    --track-dir "$TRACK_DIR" \
    --output-dir "evidence_results"

echo "============================================================"
echo " PIPELINE COMPLETE"
echo "============================================================"
echo "Check evidence_results/evidence_report.json for the WJF, p50, and p90 delta metrics."
