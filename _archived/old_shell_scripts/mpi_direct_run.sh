#!/bin/bash
#SBATCH --job-name=mpi_direct_run
# Execute the Python script and capture its output
module load python
module load conda
conda activate 2DSims
export $(python user_inputs_to_shell.py)

# # USER_INPUTS
# echo $USER_allocation
# echo $USER_JobName
# echo $USER_walltime
# echo $USER_nodes
# echo $USER_queue
# echo $USER_email

#SBATCH --job-name=$USER_JobName
#SBATCH -A $USER_allocation
#SBATCH -t $USER_walltime
#SBATCH --nodes=$USER_nodes
#SBATCH --output=NERSC/output/job_outputs/job_output_%j_$USER_JobName.txt
#SBATCH --error=NERSC/output/job_outputs/job_error_%j_$USER_JobName.txt
#SBATCH --mail-user=$USER_email
#SBATCH --mail-type=ALL

##Custom SLURM Options
#SBATCH -q $USER_queue
#SBATCH -C cpu

#export OMP_NUM_THREADS=8
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export KMP_AFFINITY=verbose
export FI_LOG_LEVEL=debug

#
# cray-mpich and cray-libsci conflict with openmpi so unload them
#
module unload cray-mpich
module unload cray-libsci
module use /global/common/software/m3169/perlmutter/modulefiles
module load openmpi

#conda activate 2DSims

# Check if output directory exists, create if not
mkdir -p NERSC/output/job_outputs

python3 NERSC/batchRun.py $USER_JobName 30