#!/bin/bash

#How to run
# $bash run_multiple_sbatch.sh cassio
# $bash run_multiple_sbatch.sh prince
if [ $# -lt 1 ]; then
  echo "Need more parameter"
  exit 2
fi

cluster=$1
datasets=(diginetica yoochoose1_4 yoochoose1_64)

for d in ${datasets[@]}; do
  if [ ${cluster} == "cassio" ]; then
    sbatch --export=dataset=$d slurm_jobs_cassio.sbatch
  elif [ ${cluster} == "prince" ]; then
    sbatch --export=dataset=$d slurm_jobs_prince.sbatch
  fi
done
