#!/bin/bash

# Define job name and argument in one place
#JOB_NAME="job_output_test"

# SLURM directives
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH --ntasks-per-node 32
#SBATCH --cpus-per-task 4
#SBATCH --output=job_output_int.txt
#SBATCH --error=job_error_int.txt

module load python
module load conda
module load openmpi
conda activate 2DSims

# OpenMP settings
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_DISPLAY_AFFINITY=true
#export OMP_AFFINITY_FORMAT="host=%H, pid=%P, thread_num=%n, thread affinity=%A"

# Check if output directory exists, create if not
#mkdir -p NERSC/output/job_outputs

# Execute the application and capture all output
python3 NERSC/batchRun.py core_test 1