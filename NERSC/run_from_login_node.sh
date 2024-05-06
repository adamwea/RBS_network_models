#!/bin/bash
#SBATCH --job-name=debug_node_run
#SBATCH -A m2043
#SBATCH -t 00:05:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=256
#SBATCH --output=NERSC/output/job_outputs/job_%j_output_debug_node_run.txt
#SBATCH --error=NERSC/output/job_outputs/job_%j_error_debug_node_run.txt
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH --exclusive
    
#bash NERSC/neuron_check.sh

#bash NERSC/neuron_run.sh
module load conda
conda activate 2DSims_nersc
module load openmpi
export OMP_NUM_THREADS=1

mkdir -p NERSC/output/job_outputs

# change directory
cd NERSC

#check the versions of the installed software
nrniv --version
mpiexec --version
mpicc --version
echo "Running the simulation"

#get jobname from sbatch
JOB_NAME=$SLURM_JOB_NAME
#get q name from sbatch
Q_NAME=$SLURM_JOB_QOS

#run the simulation
# Number of nodes
NODES=${SLURM_NNODES}
# Processes per node
PPN=${SLURM_NTASKS_PER_NODE}
# Total number of processes
NP=$((NODES * PPN))
# Run the MPI command
echo "mpiexec --display-map --map-by ppr:1:core -np ${NP} nrniv -mpi batchRun.py debug_node_run 10"
#mpiexec --display-map --map-by ppr:1:core -np ${NP} nrniv -mpi batchRun.py debug_node_run 15