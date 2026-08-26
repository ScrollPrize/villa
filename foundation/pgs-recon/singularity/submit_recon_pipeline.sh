#!/bin/bash

#SBATCH -A col_seales_uksr
#SBATCH --mail-type=END
#SBATCH --job-name=pgs-recon
#SBATCH --output=pgs-recon_%A_out.txt
#SBATCH --time=1-00:00:00

# Make rclone available on the container and tell system where to look
#SBATCH --export=SINGULARITY_BIND='/share/singularity/bin',SINGULARITYENV_PREPEND_PATH='/share/singularity/bin'

if [[ $# -lt 1 ]]; then
    echo "Usage: submit_recon_pipeline.sh <dtn_path_to_input_folder> <pgs-recon args>"
    echo "  Path should be a path accessible from dtn."
    echo "  Example: submit_recon_pipeline.sh /mnt/gemini1-4/seales_uksr/herculaneum/Dailies/Preinterval/20211207_103351_7Dec2021_PHercPHerc1186Cr01_90d8c473"
    echo "Usage example with sbatch:"
    echo "  sbatch -p SKY32M192_L --cpus-per-task=32 submit_single_recon.sh /mnt/gemini1-4/seales_uksr/herculaneum/Dailies/Preinterval/20211207_103351_7Dec2021_PHercPHerc1186Cr01_90d8c473"
    echo "Usage example with env variables:"
    echo "  sbatch --export ALL,PGS_EXPOSURE=1.5 submit_recon_pipeline.sh /mnt/gemini1-4/seales_uksr/herculaneum/Dailies/Preinterval/20211207_103351_7Dec2021_PHercPHerc1186Cr01_90d8c473"
    exit 2
fi

# Exit if any command fails
set -e

# Check if "dtn" configured as ssh host
if ! ssh dtn exit ; then
  echo "ssh Host 'dtn' not found or connection unsuccessful. Please add Host 'dtn' to your ssh config"
  exit 2
fi

# Make sure overlay files exist
for overlay in "pgs-recon.overlay" "registration-toolkit.overlay"
do
  if ! test -f "${overlay}" ; then
    echo "Creating ${overlay}"
    dd if=/dev/zero of="${overlay}" bs=1M count=500 && mkfs.ext3 -F "${overlay}"
  fi
done

# Setup
module load ccs/singularity

# Important paths
containers_dir="${PROJECT}/seales_uksr/containers"
pgs_recon_container="${containers_dir}/pgs-recon.sif"
registration_toolkit_container="${containers_dir}/registration-toolkit.sif"
root_processing_dir="${PSCRATCH}/seales_uksr/herculaneum-processing"
recon_config_file="${PGS_CONFIG:-"${PSCRATCH}/seales_uksr/dri-experiments-drive/2021-pgs-recon/configs/pgs-recon-import-global.txt"}"
processed_dir="/mnt/gemini1-4/seales_uksr/herculaneum/Processed"

# Convert options
exposure=${PGS_EXPOSURE:-"2.5"}
shadows=${PGS_SHADOWS:-"50"}

# Generated options
dtn_path="$1"
job_name="$( basename "${dtn_path}" )"
processed_job_name="${job_name}_processed_$(date +'%Y%m%d_%H%M%S')"
processed_job_dir="${root_processing_dir}/${processed_job_name}"

# Copy data to LCC
echo "Copying dataset to lcc"
time rsync -a dtn:"${dtn_path%/}" "${root_processing_dir}/"

# Convert raw images to JPG
echo "Converting raw images to jpg"
time singularity run --overlay pgs-recon.overlay "${pgs_recon_container}" \
  pgs-convert -i "${root_processing_dir}/${job_name}" \
  --exposure="${exposure}" --shadows="${shadows}" \
  -o "${processed_job_dir}/jpg"

# Run reconstruction
echo "Starting reconstruction"
time singularity run --overlay pgs-recon.overlay "${pgs_recon_container}" \
  pgs-recon -c "${recon_config_file}" \
  -i "${processed_job_dir}/jpg" \
  -o "${processed_job_dir}" \
  -n "${job_name}" \
  "${@:2}"

# Reorder texture
echo "Reordering mesh"
time singularity run --overlay registration-toolkit.overlay "${registration_toolkit_container}" \
  rt_reorder_texture -i "${processed_job_dir}/mvs/${job_name}.obj" \
  -o "${processed_job_dir}/mvs/${job_name}_reordered.obj"

# Center mesh
echo "Centering and scaling mesh"
time singularity run --overlay pgs-recon.overlay "${pgs_recon_container}" \
  pgs-center \
  -i "${processed_job_dir}/mvs/${job_name}_reordered.obj" \
  -o "${processed_job_dir}/${job_name}_final.obj"

# Copy out OpenMVS final output
echo "Collecting results"
cp "${processed_job_dir}/mvs/${job_name}.obj" \
  "${processed_job_dir}/mvs/${job_name}.mtl" \
  "${processed_job_dir}/mvs/${job_name}"_material_*.jpg \
  "${processed_job_dir}/"

# Package the intermediate files (remove them as we go)
time tar -cvzf "${processed_job_dir}/intermediate.tar.gz" \
  "${processed_job_dir}/jpg" \
  "${processed_job_dir}/mvg" \
  "${processed_job_dir}/mvs" \
  "${processed_job_dir}/metadata.json" \
  $processed_job_dir/*.txt \
  --remove-files

# Copy back to Gemini in Processed/ folder
echo "Copying results to gemini"
time rsync -a --remove-source-files "${processed_job_dir}" dtn:"${processed_dir}/"

# Remove empty processing directory and copied raw files
echo "Removing working directory"
rm -rf "${processed_job_dir}" "${root_processing_dir}/${job_name}"

echo "Done."