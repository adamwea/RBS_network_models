#!/bin/bash
#SBATCH --job-name=unused_cores_test_1x8
#SBATCH -A m2043
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --output=NERSC/output/job_outputs/job_output_%j_unused_cores_test_1x8.txt
#SBATCH --error=NERSC/output/job_outputs/job_error_%j_unused_cores_test_1x8.txt
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH --exclusive

export OMP_PROC_BIND=spread
export KMP_AFFINITY=verbose
export FI_LOG_LEVEL=debug

module load python
module load conda
module unload cray-mpich
module unload cray-libsci
module use /global/common/software/m3169/perlmutter/modulefiles
module load openmpi

conda activate 2DSims

mkdir -p NERSC/output/job_outputs

python3 NERSC/batchRun.py unused_cores_test_1x8 30
