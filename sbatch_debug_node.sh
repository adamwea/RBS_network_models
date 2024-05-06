#!/bin/bash
#SBATCH --job-name=overnightRun
#SBATCH -A m2043
#SBATCH -t 00:05:00
#SBATCH -N 2
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH --exclusive
#SBATCH --output=./NERSC/output/latest_job_init_error.txt
#SBATCH --error=./NERSC/output/latest_job_init_output.txt
    
#specify the duration of the simulation and the label of the batch run
Duration_Seconds=12
Batch_Run_Label=$SLURM_JOB_NAME

#initialize
module load conda
conda activate 2DSims_nersc
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
module load openmpi
# mkdir -p NERSC/output/job_outputs
# nrniv --version #check the versions of the installed software
# mpiexec --version
# mpicc --version

# prepare batch run
export PYTHONWARNINGS="ignore:DeprecationWarning" 
run_path=$(python3 NERSC/USER_init_new_batch.py ${Batch_Run_Label}) # Initialize the batch file and store the return value in run_path
echo "Run path: ${run_path}"
JOB_ID=$SLURM_JOB_ID
# export OMP_NUM_THREADS=1
# NODES=${SLURM_NNODES} # Number of nodes (automatically retrieved from the SLURM environment variable)
# PPN=${SLURM_NTASKS_PER_NODE} # Processes per node (set by SLURM ntasks-per-node)
# PPN=128
# NP=$((NODES * PPN)) # Total number of processes

#save copy of this batch_script in the run_path
full_path=$(realpath $0)
cp ${full_path} ${run_path}/sbatch_debug_node.sh

# Run the MPI command
cd NERSC # Change directory
echo "Running the simulation"
mpiexec --display-map --map-by ppr:1:core \
nrniv -mpi batchRun.py ${run_path} ${Duration_Seconds} \
> ${run_path}/job_${JOB_ID}_mpi_output.txt \
2> ${run_path}/job_${JOB_ID}_mpi_error.txt
# #get jobname from sbatch
# JOB_NAME=$SLURM_JOB_NAME
# #get q name from sbatch
# Q_NAME=$SLURM_JOB_QOS
# #run the simulation
# # Number of nodes
# NODES=${SLURM_NNODES}
# # Processes per node
# PPN=${SLURM_NTASKS_PER_NODE}
# # Total number of processes
# NP=$((NODES * PPN))
# Run the MPI command
