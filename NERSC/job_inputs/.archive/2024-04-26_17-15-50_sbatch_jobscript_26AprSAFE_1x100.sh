#!/bin/bash
#SBATCH --job-name=26AprSAFE_1x100
#SBATCH -A m2043
#SBATCH -t 15:00:00
#SBATCH --nodes=1
#SBATCH --output=NERSC/output/job_outputs/job_output_%j_26AprSAFE_1x100.txt
#SBATCH --error=NERSC/output/job_outputs/job_error_%j_26AprSAFE_1x100.txt
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --exclusive
    
module load conda
conda activate 2DSims
module load openmpi
mkdir -p NERSC/output/job_outputs

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
cd NERSC
mpiexec --use-hwthread-cpus -np 100 -bind-to hwthread nrniv -mpi batchRun.py 26AprSAFE_1x100 5
    