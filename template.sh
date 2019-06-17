#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --ntasks-per-node=4
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --exclude=compute-0-[EXCLUDE_COMPUTE]
#SBATCH -o [LOGFILE]
#SBATCH -e [LOGFILE]
set -x
set -u
set -e
module load singularity
time [CUDADEVICE] \
	singularity exec --nv [SINGULARITY_IMG] \
	python [SINGULARITY_SCRIPT] \
[JOB_PARAMS]

