#!/bin/bash
#SBATCH --job-name=mpi_direct_test
#SBATCH -A m2043
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --output=NERSC/output/job_outputs/job_output_%j_mpi_direct_test.txt
#SBATCH --error=NERSC/output/job_outputs/job_error_%j_mpi_direct_test.txt
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL

##Custom SLURM Options
#SBATCH -q debug
#SBATCH -C cpu

module load python
module load conda

#
# cray-mpich and cray-libsci conflict with openmpi so unload them
#
module unload cray-mpich
module unload cray-libsci
module use /global/common/software/m3169/perlmutter/modulefiles
module load openmpi

conda activate 2DSims

# Check if output directory exists, create if not
mkdir -p NERSC/output/job_outputs

python3 NERSC/batchRun.py mpi_direct_test 30