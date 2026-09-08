#!/bin/bash

#SBATCH -A col_seales_uksr
#SBATCH --mail-type=END
#SBATCH --job-name=pgs-recon
#SBATCH --output=pgs-recon_%A_%a_out.txt

# Make rclone available on the container and tell system where to look
#SBATCH --export=SINGULARITY_BIND='/share/singularity/bin',SINGULARITYENV_PREPEND_PATH='/share/singularity/bin'

# Make sure overlay files exist
for overlay in "pgs-recon.overlay"
do
  if ! test -f "${overlay}" ; then
    echo "Creating ${overlay}"
    dd if=/dev/zero of="${overlay}" bs=1M count=500 && mkfs.ext3 -F "${overlay}"
  fi
done

module load ccs/singularity

time singularity run --overlay pgs-recon.overlay ${PROJECT}/seales_uksr/containers/pgs-recon.sif "$@"