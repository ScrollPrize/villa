#!/bin/bash

cnt=0
for i in $(cat "$1"); do 
  if grep -Fxq "$i" "submitted.txt"; then
    continue
  fi

  # Check if we've already submitted 10 jobs
  if (($cnt == 10)); then
     echo "Submitted ${cnt} jobs. Delaying 2 mins to avoid overloading ssh..."
     sleep 120
     echo "Resuming..."
     echo
     cnt=0
  fi

  # Submit a new job
  echo "Submitting ${i}..."
  sbatch -p CAL48M192_L --cpus-per-task=32 --mem=0 --export ALL,PGS_EXPOSURE=2,PGS_SHADOWS=40 submit_recon_pipeline.sh "/mnt/gemini1-4/seales_uksr/herculaneum/Dailies/${i}/"
  echo ${i} >> submitted.txt
  echo

  cnt=$(($cnt+1))
done

