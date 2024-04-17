#!/bin/bash

# Define job name and argument in one place
#JOB_NAME="job_output_test"

# SLURM directives
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J 2node_30sec_test
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node 32
#SBATCH --cpus-per-task 8
#SBATCH --output=NERSC/output/job_outputs/job_output_%j_2node_30sec_test.txt
#SBATCH --error=NERSC/output/job_outputs/job_error_%j_2node_30sec_test.txt

module load python
module load conda
module load openmpi
conda activate 2DSims

# OpenMP settings
export OMP_NUM_THREADS=8
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export OMP_DISPLAY_AFFINITY=true
export OMP_AFFINITY_FORMAT="host=%H, pid=%P, thread_num=%n, thread affinity=%A"

# Check if output directory exists, create if not
mkdir -p NERSC/output/job_outputs

#check alloc
srun -n 32 -c 8 --cpu-bind=cores check-hybrid.gnu.pm |sort -k4

# Execute the application and capture all output
#python3 NERSC/batchRun.py 2node_30sec_test 30