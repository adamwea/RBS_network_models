#!/bin/bash
#SBATCH --job-name=128popevol_1x128
#SBATCH -A m2043
#SBATCH -t 00:05:00
#SBATCH --nodes=1
#SBATCH --output=NERSC/output/job_outputs/job_output_%j_128popevol_1x128.txt
#SBATCH --error=NERSC/output/job_outputs/job_error_%j_128popevol_1x128.txt
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug
#SBATCH -C cpu
    
module load python
module load conda
module load openmpi
conda activate 2DSims
mkdir -p NERSC/output/job_outputs
cd NERSC
mpiexec -np 128 nrniv -mpi batchRun.py 128popevol_1x128 5
    