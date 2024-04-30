#!/bin/bash
#SBATCH --job-name=debug_node_run
#SBATCH -A m2043
#SBATCH -t 09:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --output=NERSC/output/job_outputs/job_%j_output_debug_node_run.txt
#SBATCH --error=NERSC/output/job_outputs/job_%j_error_debug_node_run.txt
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --exclusive

module load conda
conda activate neuron_env
module load openmpi
#module load cmake

mkdir -p NERSC/output/job_outputs

# Set the library path for NEURON and Python libraries
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/neuron/lib:$LD_LIBRARY_PATH

# Set the path to include the NEURON binaries
export PATH=$HOME/neuron/bin:$PATH

# Number of nodes (automatically retrieved from the SLURM environment variable)
NODES=${SLURM_NNODES}
# Processes per node (set by SLURM ntasks-per-node)
PPN=${SLURM_NTASKS_PER_NODE}
# Total number of processes
NP=$((NODES * PPN))

# Change directory
cd NERSC
#check the versions of the installed software
nrniv --version
mpiexec --version
mpicc --version
echo "Running the simulation"
# Run the MPI command
mpiexec --display-map --map-by ppr:1:core -np ${NP} nrniv -mpi batchRun.py debug_node_run 10
