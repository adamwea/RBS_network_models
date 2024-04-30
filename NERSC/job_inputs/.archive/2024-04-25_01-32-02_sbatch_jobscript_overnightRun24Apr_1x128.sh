#!/bin/bash
#SBATCH --job-name=overnightRun24Apr_1x128
#SBATCH -A m2043
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --output=NERSC/output/job_outputs/job_output_%j_overnightRun24Apr_1x128.txt
#SBATCH --error=NERSC/output/job_outputs/job_error_%j_overnightRun24Apr_1x128.txt
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH --exclusive
    
module load conda
conda activate 2DSims
module load openmpi
mkdir -p NERSC/output/job_outputs

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMPI_MCA_pml=cm
export OMPI_MCA_mtl=ofi
cd NERSC
mpiexec --use-hwthread-cpus -np 128 -bind-to hwthread nrniv -mpi batchRun.py overnightRun24Apr_1x128 5
    