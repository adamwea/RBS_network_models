#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH -N 4                  # Number of nodes
#SBATCH --ntasks-per-node=16  # Number of tasks per node
#SBATCH -t 2:00:00            # Time for the job (hh:mm:ss)
#SBATCH --qos=debug           # Queue, options are debug, regular, interactive
#SBATCH -C cpu                # Constraint for CPU nodes
#SBATCH --output=job_output_%j.txt # Standard output file
#SBATCH --error=job_error_%j.txt # Standard error output file

module load python
module load conda
module load openmpi

conda activate 2DSims

python3 NERSC/batchRun.py 2hr_debug_test 30