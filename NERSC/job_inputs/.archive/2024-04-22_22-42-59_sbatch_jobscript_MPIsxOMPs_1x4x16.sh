#!/bin/bash
#SBATCH --job-name=MPIsxOMPs_1x4x16
#SBATCH -A m2043
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --output=NERSC/output/job_outputs/job_output_%j_MPIsxOMPs_1x4x16.txt
#SBATCH --error=NERSC/output/job_outputs/job_error_%j_MPIsxOMPs_1x4x16.txt
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q regular
#SBATCH -C cpu

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

python3 NERSC/batchRun.py MPIsxOMPs_1x4x16 30
